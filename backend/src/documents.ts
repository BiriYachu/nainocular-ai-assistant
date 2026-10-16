// DOCUMENT PROCESS: everything that happens when a file is uploaded.
//   1. check the file   2. extract the text   3. split it into chunks
//   4. create embeddings   5. save the text in MongoDB and the embeddings in Qdrant (the vector database)
import { createHash, randomUUID } from "crypto";
import { PDFParse } from "pdf-parse";
import { generateEmbeddings } from "./ai";
import {
  countDocuments,
  deleteDocument,
  deleteFailedDocuments,
  discardDocument,
  findDocument,
  findProcessedDocumentByFingerprint,
  isDuplicateKeyError,
  markDocumentFailed,
  markDocumentProcessed,
  publicDocument,
  saveChunks,
  saveNewDocument,
} from "./database";
import type { Chunk, FileType } from "./database";
import { AppError, fixFilenameEncoding, numberSetting, sanitizeFilename } from "./helpers";
import { deleteDocumentVectors, saveVectors } from "./vectors";

// Settings you can change.
export const CHUNK_SIZE = 800; // maximum characters in one chunk
export const CHUNK_OVERLAP = 100; // characters shared between neighbouring chunks

// The text of a document. For PDFs "pages" holds the text of each page that HAS text (so chunks get
// page numbers) and "totalPages" is how many pages the PDF has. For TXT files both are empty / 0,
// because text files have no pages.
export type ExtractedText = {
  text: string;
  pages: { pageNumber: number; text: string }[];
  totalPages: number;
};

// ---------- Step 1: check the file ----------

// A real PDF starts with the characters "%PDF-". We check the content, not only the name.
function looksLikePdf(buffer: Buffer): boolean {
  return buffer.subarray(0, 5).toString("latin1") === "%PDF-";
}

// Some text files are saved as UTF-16 (older Windows Notepad called it "Unicode"). They start with a
// byte-order mark (FF FE or FE FF) and contain zero bytes on purpose.
function startsWithUtf16Mark(buffer: Buffer): boolean {
  return buffer.length >= 2 && ((buffer[0] === 0xff && buffer[1] === 0xfe) || (buffer[0] === 0xfe && buffer[1] === 0xff));
}

// Plain text never contains zero bytes. Pictures, zip files and Word files do.
function looksLikeText(buffer: Buffer): boolean {
  if (startsWithUtf16Mark(buffer)) {
    return true;
  }
  const checkLength = Math.min(buffer.length, 8000);
  for (let i = 0; i < checkLength; i++) {
    if (buffer[i] === 0) {
      return false;
    }
  }
  return true;
}

// Decides the file type from the name AND the content. Throws if they do not agree.
export function detectFileType(filename: string, buffer: Buffer): FileType {
  const name = filename.toLowerCase();

  if (name.endsWith(".pdf")) {
    if (!looksLikePdf(buffer)) {
      throw new AppError("This file has a .pdf name but is not a valid PDF.", 400);
    }
    return "pdf";
  }
  if (name.endsWith(".txt")) {
    if (looksLikePdf(buffer) || !looksLikeText(buffer)) {
      throw new AppError("This file has a .txt name but is not a plain text file.", 400);
    }
    return "txt";
  }
  throw new AppError("Unsupported file type. Please upload a PDF or TXT file.", 400);
}

// ---------- Step 2: extract the text ----------

// Tidies text so it is easier to search and to quote:
// - "normalize" turns special characters into plain ones (the ligature "ﬁ" becomes "fi",
//   a non-breaking space becomes a normal space)
// - invisible characters (soft hyphens, zero-width spaces) are removed
// - blank lines between paragraphs are kept
function cleanText(text: string): string {
  return text
    .normalize("NFKC")
    .replace(/[­​-‍﻿]/g, "") // invisible characters
    .replace(/\r\n/g, "\n") // Windows line endings -> normal
    .replace(/[ \t]+/g, " ") // many spaces -> one space
    .replace(/\n{3,}/g, "\n\n") // many blank lines -> one blank line
    .trim();
}

async function extractTextFromPdf(buffer: Buffer): Promise<ExtractedText> {
  const parser = new PDFParse({ data: new Uint8Array(buffer) });
  const pages: { pageNumber: number; text: string }[] = [];
  let totalPages = 0;

  try {
    const result = await parser.getText(); // result.pages = [{ num: 1, text: "..." }, ...]
    totalPages = result.pages.length;
    for (const page of result.pages) {
      const text = cleanText(page.text);
      if (text !== "") {
        pages.push({ pageNumber: page.num, text: text }); // pages without text are skipped
      }
    }
  } catch (error) {
    console.error("PDF parsing failed:", error instanceof Error ? error.name : "unknown error");
    throw new AppError("The PDF could not be read. It may be corrupted or password-protected.", 422);
  } finally {
    await parser.destroy(); // free the memory used by the parser
  }

  if (pages.length === 0) {
    throw new AppError(
      "No readable text was found in this PDF. It may be a scanned document. OCR is not supported yet.",
      422
    );
  }
  return { text: pages.map((page) => page.text).join("\n\n"), pages: pages, totalPages: totalPages };
}

// Turns the bytes of a text file into text. Which "alphabet code" (encoding) was used?
//  - UTF-16: the file starts with a byte-order mark
//  - UTF-8: the normal case today
//  - otherwise: an old Windows file ("windows-1252"), for example "café" saved by an old program
function decodeText(buffer: Buffer): string {
  if (buffer.length >= 2 && buffer[0] === 0xff && buffer[1] === 0xfe) {
    return new TextDecoder("utf-16le").decode(buffer);
  }
  if (buffer.length >= 2 && buffer[0] === 0xfe && buffer[1] === 0xff) {
    return new TextDecoder("utf-16be").decode(buffer);
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer); // "fatal": throw if the bytes are not valid UTF-8
  } catch {
    return new TextDecoder("windows-1252").decode(buffer);
  }
}

function extractTextFromTxt(buffer: Buffer): ExtractedText {
  // (cleanText also removes a leading BOM, a hidden marker some editors add.)
  const text = cleanText(decodeText(buffer));
  if (text === "") {
    throw new AppError("The text file is empty or contains only whitespace.", 400);
  }
  return { text: text, pages: [], totalPages: 0 };
}

export async function extractText(fileType: FileType, buffer: Buffer): Promise<ExtractedText> {
  if (fileType === "pdf") {
    return extractTextFromPdf(buffer);
  }
  return extractTextFromTxt(buffer);
}

// ---------- Step 3: split the text into chunks ----------

// Cuts one long text into pieces. Example: size 10, overlap 3, "ABCDEFGHIJKLMNOPQRS" gives
//   "ABCDEFGHIJ", "HIJKLMNOPQ", "OPQRS"   (each piece starts 7 characters after the previous one)
// The overlap makes sure a sentence cut at the end of one piece also appears whole in the next.
export function cutText(text: string, size: number, overlap: number): string[] {
  const step = size - overlap; // how far we move forward each time
  if (step <= 0) {
    throw new Error("Chunk overlap must be smaller than chunk size.");
  }

  const pieces: string[] = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + size, text.length);
    const piece = text.slice(start, end).trim();
    if (piece !== "") {
      pieces.push(piece); // never keep empty pieces
    }
    if (end === text.length) {
      break; // reached the end of the text
    }
    start = start + step;
  }
  return pieces;
}

// Turns the extracted text into Chunk objects, in order, each with its own ID.
export function splitIntoChunks(
  documentId: string,
  filename: string,
  extracted: ExtractedText,
  size: number = CHUNK_SIZE,
  overlap: number = CHUNK_OVERLAP
): Chunk[] {
  const chunks: Chunk[] = [];

  // PDF: cut each page separately, so every chunk has a reliable page number.
  // TXT: no pages, so we cut the whole text and never invent page numbers.
  const parts = extracted.pages.length > 0 ? extracted.pages : [{ pageNumber: undefined, text: extracted.text }];

  for (const part of parts) {
    for (const piece of cutText(part.text, size, overlap)) {
      const chunk: Chunk = {
        id: randomUUID(),
        documentId: documentId,
        chunkIndex: chunks.length, // 0, 1, 2, ... in order
        text: piece,
        filename: filename,
      };
      if (part.pageNumber !== undefined) {
        chunk.pageNumber = part.pageNumber; // PDFs only. For TXT the field simply does not exist.
      }
      chunks.push(chunk);
    }
  }
  return chunks;
}

// ---------- The whole upload workflow ----------

// A short note for the user when a PDF had pages without text (for example scanned pages).
function makeWarning(extracted: ExtractedText): string | undefined {
  const skippedPages = extracted.totalPages - extracted.pages.length;
  if (skippedPages <= 0) {
    return undefined;
  }
  return skippedPages + " of " + extracted.totalPages + " pages had no readable text (maybe scanned) and were skipped.";
}

// Runs all the steps above for one user ("ownerId"), saves the result in MongoDB, and returns the saved document.
// "file" is what Multer gives us: the original name and the file bytes (kept in memory, never on disk).
export async function uploadDocument(ownerId: string, file: { originalname: string; buffer: Buffer }) {
  if (file.buffer.length === 0) {
    throw new AppError("The uploaded file is empty.", 400);
  }

  // Check the file. The filename is repaired and cleaned, and the CONTENT must match the file type.
  const filename = sanitizeFilename(fixFilenameEncoding(file.originalname));
  const fileType = detectFileType(filename, file.buffer);

  // Did THIS user already process this exact file? Then reuse it. (A fingerprint = a hash of the file bytes + name.)
  const fingerprint = createHash("sha256").update(file.buffer).digest("hex") + "|" + filename;
  const existing = await findProcessedDocumentByFingerprint(ownerId, fingerprint);
  if (existing) {
    return { alreadyProcessed: true, document: publicDocument(existing) };
  }
  await deleteFailedDocuments(ownerId); // forget this user's older failed uploads

  // Every user may keep a limited number of documents.
  const maxDocuments = numberSetting("MAX_DOCUMENTS_PER_USER", 20);
  if ((await countDocuments(ownerId)) >= maxDocuments) {
    throw new AppError("You already have " + maxDocuments + " documents. Delete one before uploading another.", 409);
  }

  // Create the document record. Its status stays "processing" until every step has worked.
  const documentId = randomUUID();
  await saveNewDocument({
    id: documentId,
    ownerId: ownerId,
    filename: filename,
    fileType: fileType,
    fingerprint: fingerprint,
    status: "processing",
    characterCount: 0,
    chunkCount: 0,
    uploadedAt: new Date(),
  });

  try {
    const extracted = await extractText(fileType, file.buffer);
    const chunks = splitIntoChunks(documentId, filename, extracted);

    // A huge document would need thousands of embedding calls. Refuse it before spending any money.
    const maxChunks = numberSetting("MAX_CHUNKS_PER_DOCUMENT", 500);
    if (chunks.length > maxChunks) {
      throw new AppError(
        "This document is too long (" + chunks.length + " chunks, the limit is " + maxChunks + "). Please upload a shorter one.",
        413
      );
    }

    const embeddings = await generateEmbeddings(chunks.map((chunk) => chunk.text));

    // Save the text in MongoDB and the embeddings in Qdrant, THEN mark the document as processed.
    await saveChunks(documentId, chunks);
    await saveVectors(chunks.map((chunk, index) => ({ id: chunk.id, ownerId: ownerId, documentId: documentId, embedding: embeddings[index] })));
    await markDocumentProcessed(documentId, extracted.text.length, chunks.length, makeWarning(extracted));
  } catch (error) {
    // The same user uploaded the same file twice at the same moment. The database allowed only ONE
    // of them to become "processed". We throw this copy away and return the winner.
    if (isDuplicateKeyError(error)) {
      await discardDocument(documentId);
      await deleteDocumentVectors(documentId).catch(() => {});
      const winner = await findProcessedDocumentByFingerprint(ownerId, fingerprint);
      if (winner) {
        return { alreadyProcessed: true, document: publicDocument(winner) };
      }
    }

    // Something failed: remove partial chunks and vectors, and keep only a SAFE reason on the document.
    const safeMessage = error instanceof AppError ? error.message : "Processing failed unexpectedly.";
    await markDocumentFailed(documentId, safeMessage).catch(() => console.error("Could not clean up " + documentId));
    await deleteDocumentVectors(documentId).catch(() => console.error("Could not remove the vectors of " + documentId));
    throw error; // the server sends a short, safe message to the browser
  }

  const savedDocument = await findDocument(ownerId, documentId);
  if (savedDocument === null) {
    throw new AppError("The document could not be saved.", 500);
  }
  return { alreadyProcessed: false, document: publicDocument(savedDocument) };
}

// Deletes a document completely: its chats and chunks (MongoDB) and its vectors (Qdrant).
// Returns "deleted", "not-found", or "busy" (still being processed).
export async function removeDocument(ownerId: string, documentId: string) {
  const result = await deleteDocument(ownerId, documentId);
  if (result === "deleted") {
    // If Qdrant is down right now, the vectors stay behind. That is harmless: no document points to them,
    // and every search is limited to documents that exist.
    await deleteDocumentVectors(documentId).catch(() => console.error("Could not remove the vectors of " + documentId));
  }
  return result;
}
