// Simple checks for the chunking code. Run with: npm run test:chunking
import { cutText, splitIntoChunks } from "../src/documents";

let failures = 0;
function check(name: string, condition: boolean) {
  console.log((condition ? "PASS " : "FAIL ") + name);
  if (!condition) failures++;
}

// Example from the comments: size 10, overlap 3.
const pieces = cutText("ABCDEFGHIJKLMNOPQRS", 10, 3);
check("splits with overlap", JSON.stringify(pieces) === JSON.stringify(["ABCDEFGHIJ", "HIJKLMNOPQ", "OPQRS"]));
check("overlap is 3 characters", pieces[0].slice(-3) === pieces[1].slice(0, 3));

check("short text gives one chunk", cutText("hello", 800, 100).length === 1);
check("empty text gives no chunks", cutText("   ", 800, 100).length === 0);

const longText = "a".repeat(2500);
const longPieces = cutText(longText, 800, 100);
check("no chunk is longer than 800", longPieces.every((p) => p.length <= 800));
check("2500 chars -> 4 chunks", longPieces.length === 4);

// Chunks keep order and page numbers.
const pdfChunks = splitIntoChunks("doc1", "a.pdf", {
  text: "",
  pages: [
    { pageNumber: 1, text: "x".repeat(900) },
    { pageNumber: 2, text: "short page" },
  ],
});
check("PDF chunk indexes are 0..n-1", pdfChunks.map((c) => c.chunkIndex).join() === "0,1,2");
check("PDF page numbers kept in order", pdfChunks.map((c) => c.pageNumber).join() === "1,1,2");

const txtChunks = splitIntoChunks("doc2", "a.txt", { text: "hello world", pages: [] });
check("TXT has no page number", txtChunks[0].pageNumber === undefined);

console.log(failures === 0 ? "\nAll chunking checks passed." : "\n" + failures + " check(s) failed.");
process.exit(failures === 0 ? 0 : 1);
