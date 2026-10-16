// Everything that talks to MongoDB lives in this file.
//
// Four collections (think: four tables):
//   documents      one record per uploaded file (name, status, counts)
//   chunks         the small text pieces of each document (the embeddings are NOT here: they live in Qdrant,
//                  see vectors.ts)
//   conversations  one record per chat
//   messages       the questions and answers of each chat
//
// Every document and conversation has an "ownerId" (a hash of the user's private key).
// Every function below that reads or changes them needs the ownerId, so one user can
// never see or change another user's data.
import { randomUUID } from "crypto";
import { MongoClient } from "mongodb";
import type { Db } from "mongodb";
import { AppError } from "./helpers";

// ---------- Types: the shape of the data we store ----------

export type FileType = "pdf" | "txt";

// processing = work in progress, processed = ready for questions, failed = something went wrong
export type DocumentStatus = "processing" | "processed" | "failed";

export type DocumentRecord = {
  id: string;
  ownerId: string;
  filename: string;
  fileType: FileType;
  fingerprint: string; // identifies the file contents, so the same file is not processed twice
  status: DocumentStatus;
  characterCount: number;
  chunkCount: number;
  uploadedAt: Date;
  warning?: string; // for example "2 of 8 pages had no readable text"
  errorMessage?: string; // a safe message, only when status is "failed"
};

export type Chunk = {
  id: string;
  documentId: string;
  chunkIndex: number; // 0, 1, 2, ... the order inside the document
  text: string;
  filename: string;
  pageNumber?: number; // only for PDFs ("?" means optional)
};

// Chunks saved by an older version of the app still have their embedding inside MongoDB.
// vectors.ts moves those to Qdrant when the server starts.
export type OldChunk = Chunk & { embedding?: number[] };

// A source shown to the user. It is only created AFTER the backend verified it.
export type Source = {
  documentId: string;
  filename: string;
  chunkId: string;
  chunkIndex: number;
  excerpt: string;
  pageNumber?: number;
  similarity: number;
};

// answered = normal answer, not_found = the fallback sentence, unverified = the AI's answer failed our checks
export type AnswerStatus = "answered" | "not_found" | "unverified";

export type ConversationRecord = {
  id: string;
  ownerId: string;
  documentIds: string[]; // the documents this chat is about
  title: string; // the first question, shortened
  createdAt: Date;
  updatedAt: Date;
};

export type MessageRecord = {
  id: string;
  conversationId: string;
  role: "user" | "assistant";
  text: string;
  status?: AnswerStatus; // assistant messages only
  sources: Source[];
  note?: string;
  createdAt: Date;
};

// ---------- Connecting ----------

let client: MongoClient | null = null;
let database: Db | null = null;

// Connects to MongoDB and creates the indexes. Indexes make searches fast,
// and "unique" indexes stop duplicate records.
export async function connectToDatabase(uri?: string, databaseName?: string): Promise<void> {
  client = new MongoClient(uri || process.env.MONGODB_URI || "mongodb://localhost:27017", {
    serverSelectionTimeoutMS: 5000,
  });
  await client.connect();
  database = client.db(databaseName || process.env.MONGODB_DATABASE || "nainocular");

  const collection = collections();

  // Older versions of this app created indexes that are wrong now. Remove them if they are still there.
  await collection.documents.dropIndex("contentKey_1").catch(() => {});
  await collection.documents.dropIndex("fingerprint_1").catch(() => {});
  await collection.conversations.dropIndex("documentIds_1_updatedAt_-1").catch(() => {});

  // Documents and chats saved before user separation existed have no owner. Give each one its own
  // "unowned" owner. They stay in the database, but no user can ever reach them.
  const oldDocuments = await collection.documents.find({ ownerId: { $exists: false } }, { projection: { _id: 0, id: 1 } }).toArray();
  for (const oldDocument of oldDocuments) {
    await collection.documents.updateOne({ id: oldDocument.id }, { $set: { ownerId: "unowned-" + oldDocument.id } });
  }
  const oldChats = await collection.conversations.find({ ownerId: { $exists: false } }, { projection: { _id: 0, id: 1 } }).toArray();
  for (const oldChat of oldChats) {
    await collection.conversations.updateOne({ id: oldChat.id }, { $set: { ownerId: "unowned-" + oldChat.id } });
  }

  await collection.documents.createIndex({ id: 1 }, { unique: true });
  await collection.documents.createIndex({ ownerId: 1, uploadedAt: -1 });
  // One PROCESSED document per user and file fingerprint. This stops two uploads of the same file at
  // the same moment from both being saved as "processed".
  await collection.documents.createIndex(
    { ownerId: 1, fingerprint: 1 },
    { unique: true, partialFilterExpression: { status: "processed" } }
  );

  await collection.chunks.createIndex({ id: 1 }, { unique: true });
  // One chunk per (document, chunk number): this is what prevents duplicate chunks.
  await collection.chunks.createIndex({ documentId: 1, chunkIndex: 1 }, { unique: true });

  await collection.conversations.createIndex({ id: 1 }, { unique: true });
  await collection.conversations.createIndex({ ownerId: 1, updatedAt: -1 });
  await collection.conversations.createIndex({ documentIds: 1 });

  await collection.messages.createIndex({ id: 1 }, { unique: true });
  await collection.messages.createIndex({ conversationId: 1, createdAt: 1 });
}

export async function closeDatabase(): Promise<void> {
  if (client) {
    await client.close();
  }
  client = null;
  database = null;
}

// Used by the health check: "is MongoDB there?"
export async function pingDatabase(): Promise<boolean> {
  try {
    await collections().documents.countDocuments({}, { limit: 1 });
    return true;
  } catch {
    return false;
  }
}

// Used by tests to remove their temporary database.
export async function dropDatabase(): Promise<void> {
  if (database) {
    await database.dropDatabase();
  }
}

// The four collections. Throws a clear error if MongoDB is not connected.
export function collections() {
  if (database === null) {
    throw new AppError("The database is not available. Please try again later.", 503);
  }
  return {
    documents: database.collection<DocumentRecord>("documents"),
    chunks: database.collection<OldChunk>("chunks"),
    conversations: database.collection<ConversationRecord>("conversations"),
    messages: database.collection<MessageRecord>("messages"),
  };
}

// Hides MongoDB's own "_id" field. We use our own "id" instead.
const HIDE_MONGO_ID = { projection: { _id: 0 } };

// MongoDB reports "this would create a duplicate" with error code 11000.
export function isDuplicateKeyError(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { code?: number }).code === 11000;
}

// ---------- Documents ----------

export async function saveNewDocument(record: DocumentRecord): Promise<void> {
  await collections().documents.insertOne({ ...record });
}

export async function findDocument(ownerId: string, id: string): Promise<DocumentRecord | null> {
  return collections().documents.findOne({ ownerId: ownerId, id: id }, HIDE_MONGO_ID);
}

// Finds several documents of this owner at once. Documents of other owners are never returned.
export async function findDocuments(ownerId: string, ids: string[]): Promise<DocumentRecord[]> {
  return collections()
    .documents.find({ ownerId: ownerId, id: { $in: ids } }, HIDE_MONGO_ID)
    .toArray();
}

// All documents of this owner, newest first. Failed uploads are not listed: the user already saw the
// error message, and they are cleaned up at the next upload.
export async function listDocuments(ownerId: string): Promise<DocumentRecord[]> {
  return collections()
    .documents.find({ ownerId: ownerId, status: { $ne: "failed" } }, HIDE_MONGO_ID)
    .sort({ uploadedAt: -1 })
    .limit(100)
    .toArray();
}

// Counts the documents that count toward the limit (failed uploads do not).
export async function countDocuments(ownerId: string): Promise<number> {
  return collections().documents.countDocuments({ ownerId: ownerId, status: { $ne: "failed" } });
}

// Only finished documents count as "this file was already processed".
export async function findProcessedDocumentByFingerprint(
  ownerId: string,
  fingerprint: string
): Promise<DocumentRecord | null> {
  return collections().documents.findOne(
    { ownerId: ownerId, fingerprint: fingerprint, status: "processed" },
    HIDE_MONGO_ID
  );
}

// Removes this owner's old failed uploads. Called before a new upload starts, so they never pile up.
export async function deleteFailedDocuments(ownerId: string): Promise<void> {
  await collections().documents.deleteMany({ ownerId: ownerId, status: "failed" });
}

export async function markDocumentProcessed(
  id: string,
  characterCount: number,
  chunkCount: number,
  warning?: string
): Promise<void> {
  const changes: Record<string, unknown> = { status: "processed", characterCount, chunkCount };
  if (warning) {
    changes.warning = warning;
  }
  await collections().documents.updateOne({ id: id }, { $set: changes, $unset: { errorMessage: "" } });
}

// A failed document keeps a safe error message and loses any half-saved chunks.
export async function markDocumentFailed(id: string, safeMessage: string): Promise<void> {
  await collections().chunks.deleteMany({ documentId: id });
  await collections().documents.updateOne({ id: id }, { $set: { status: "failed", errorMessage: safeMessage } });
}

// Throws away a document record and its chunks without any checks. Used only by the upload code
// for a copy that lost the "same file uploaded twice at once" race.
export async function discardDocument(id: string): Promise<void> {
  await collections().chunks.deleteMany({ documentId: id });
  await collections().documents.deleteOne({ id: id });
}

// Called once when the server starts. If the server stopped in the middle of processing a file,
// that document would stay "processing" forever. At startup nothing can really be processing,
// so we mark those documents as failed. Returns their ids (the caller also removes their vectors).
export async function markInterruptedDocuments(): Promise<string[]> {
  const interrupted = await collections()
    .documents.find({ status: "processing" }, { projection: { _id: 0, id: 1 } })
    .toArray();

  for (const document of interrupted) {
    await markDocumentFailed(document.id, "Processing was interrupted. Please upload the file again.");
  }
  return interrupted.map((document) => document.id);
}

// Deletes a document with its chunks, embeddings and chats.
// "busy" = it is still being processed (cannot be deleted yet), "not-found" = no such document.
export async function deleteDocument(ownerId: string, id: string): Promise<"deleted" | "not-found" | "busy"> {
  const document = await findDocument(ownerId, id);
  if (document === null) {
    return "not-found";
  }
  if (document.status === "processing") {
    return "busy";
  }

  // Chats that used this document are removed too (their answers may quote it).
  const { documents, chunks, conversations, messages } = collections();
  const chats = await conversations
    .find({ ownerId: ownerId, documentIds: id }, { projection: { _id: 0, id: 1 } })
    .toArray();
  const chatIds = chats.map((chat) => chat.id);
  await messages.deleteMany({ conversationId: { $in: chatIds } });
  await conversations.deleteMany({ ownerId: ownerId, id: { $in: chatIds } });
  await chunks.deleteMany({ documentId: id });
  await documents.deleteOne({ ownerId: ownerId, id: id });
  return "deleted";
}

// What we send to the browser (no internal fields like the fingerprint or the ownerId).
export function publicDocument(record: DocumentRecord) {
  return {
    id: record.id,
    filename: record.filename,
    fileType: record.fileType,
    characterCount: record.characterCount,
    chunkCount: record.chunkCount,
    status: record.status,
    uploadedAt: record.uploadedAt.toISOString(),
    warning: record.warning,
    errorMessage: record.errorMessage,
  };
}

// ---------- Chunks (text and details; the embeddings are in Qdrant) ----------

// Hides MongoDB's own _id and any embedding an older version left behind.
const CHUNK_FIELDS = { projection: { _id: 0, embedding: 0 } };

// Saves all chunks of a document. Older chunks of the same document are removed first,
// so processing a document again never leaves duplicates behind.
export async function saveChunks(documentId: string, chunks: Chunk[]): Promise<void> {
  await collections().chunks.deleteMany({ documentId: documentId });
  if (chunks.length > 0) {
    await collections().chunks.insertMany(chunks.map((chunk) => ({ ...chunk })));
  }
}

// Loads the chunks of one or more documents, in order.
// The caller must already have checked that the documents belong to the user.
export async function getChunks(documentIds: string[]): Promise<Chunk[]> {
  return collections()
    .chunks.find({ documentId: { $in: documentIds } }, CHUNK_FIELDS)
    .sort({ documentId: 1, chunkIndex: 1 })
    .toArray() as unknown as Promise<Chunk[]>;
}

// Loads chunks by their ids. Used after Qdrant has told us which chunks are the best matches.
export async function getChunksByIds(chunkIds: string[]): Promise<Chunk[]> {
  return collections()
    .chunks.find({ id: { $in: chunkIds } }, CHUNK_FIELDS)
    .toArray() as unknown as Promise<Chunk[]>;
}

// ----- Helpers for moving embeddings from MongoDB to Qdrant (see moveOldEmbeddings in vectors.ts) -----

// The next chunks (after the given id) that still have an embedding stored inside MongoDB.
export async function getChunksWithOldEmbeddings(afterId: string, limit: number): Promise<OldChunk[]> {
  return collections()
    .chunks.find({ embedding: { $exists: true }, id: { $gt: afterId } }, { projection: { _id: 0 } })
    .sort({ id: 1 })
    .limit(limit)
    .toArray();
}

export async function removeEmbeddingsFromChunks(chunkIds: string[]): Promise<void> {
  if (chunkIds.length > 0) {
    await collections().chunks.updateMany({ id: { $in: chunkIds } }, { $unset: { embedding: "" } });
  }
}

// Which owner does each document belong to? Returns a map: documentId -> ownerId.
export async function findOwnerIds(documentIds: string[]): Promise<Map<string, string>> {
  const documents = await collections()
    .documents.find({ id: { $in: documentIds } }, { projection: { _id: 0, id: 1, ownerId: 1 } })
    .toArray();
  return new Map(documents.map((document) => [document.id, document.ownerId]));
}

// ---------- Conversations ----------

export async function findConversation(ownerId: string, id: string): Promise<ConversationRecord | null> {
  return collections().conversations.findOne({ ownerId: ownerId, id: id }, HIDE_MONGO_ID);
}

// The most recent chats of this owner, newest first.
export async function listConversations(ownerId: string): Promise<ConversationRecord[]> {
  return collections()
    .conversations.find({ ownerId: ownerId }, HIDE_MONGO_ID)
    .sort({ updatedAt: -1 })
    .limit(50)
    .toArray();
}

// All messages of ONE conversation, oldest first.
// The caller must already have checked that the conversation belongs to the user.
export async function getMessages(conversationId: string): Promise<MessageRecord[]> {
  return collections()
    .messages.find({ conversationId: conversationId }, HIDE_MONGO_ID)
    .sort({ createdAt: 1 })
    .limit(500)
    .toArray();
}

// Returns true if the chat existed and was deleted.
export async function deleteConversation(ownerId: string, id: string): Promise<boolean> {
  const chat = await findConversation(ownerId, id);
  if (chat === null) {
    return false;
  }
  await collections().messages.deleteMany({ conversationId: id });
  await collections().conversations.deleteOne({ ownerId: ownerId, id: id });
  return true;
}

// Saves a question and its answer together. Creates the conversation if conversationId is null.
// Returns the conversation id. This is only called after a successful answer,
// so a failed AI request never leaves a half-finished chat behind.
export async function saveConversation(
  ownerId: string,
  conversationId: string | null,
  documentIds: string[],
  question: string,
  answer: { text: string; status: AnswerStatus; sources: Source[]; note?: string }
): Promise<string> {
  const now = new Date();
  let id = conversationId;

  if (id === null) {
    id = randomUUID();
    await collections().conversations.insertOne({
      id: id,
      ownerId: ownerId,
      documentIds: documentIds,
      title: Array.from(question).slice(0, 80).join(""), // the first question names the chat (cut by whole characters, so an emoji is never split)
      createdAt: now,
      updatedAt: now,
    });
  }

  const questionMessage: MessageRecord = {
    id: randomUUID(), conversationId: id, role: "user", text: question, sources: [], createdAt: now,
  };
  const answerMessage: MessageRecord = {
    id: randomUUID(), conversationId: id, role: "assistant", text: answer.text, status: answer.status,
    sources: answer.sources, note: answer.note,
    createdAt: new Date(now.getTime() + 1), // 1 ms later, so the order is always clear
  };
  await collections().messages.insertMany([questionMessage, answerMessage]);
  await collections().conversations.updateOne({ id: id }, { $set: { updatedAt: answerMessage.createdAt } });
  return id;
}
