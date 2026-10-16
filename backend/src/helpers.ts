// Small tools that several files use.
import { createHash } from "crypto";

// An error that also carries an HTTP status code (400 = bad request, 404 = not found, ...).
// The server turns it into a short JSON message for the browser.
export class AppError extends Error {
  statusCode: number;

  constructor(message: string, statusCode: number) {
    super(message);
    this.statusCode = statusCode;
  }
}

// Reads a number from backend/.env. Uses the default if it is missing or not a number.
export function numberSetting(name: string, defaultValue: number): number {
  const text = process.env[name];
  if (text === undefined || text.trim() === "") {
    return defaultValue;
  }
  const number = Number(text);
  return Number.isFinite(number) ? number : defaultValue;
}

// Every ID this app creates is a UUID like "3f2b8c1e-1234-4abc-9def-0123456789ab".
// Anything else is rejected early. This also blocks tricks like sending objects instead of text.
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isValidId(value: unknown): value is string {
  return typeof value === "string" && UUID_PATTERN.test(value);
}

// ---------- Who is the user? ----------
// There are no passwords. Each browser creates a random private key (64 hex characters) and sends it
// in the "X-User-Key" header. We never store the key itself, only a hash of it (the "owner ID").
// Every document and chat belongs to one owner ID, and the owner ID is checked on every request.

const USER_KEY_PATTERN = /^[0-9a-f]{64}$/i;

export function isValidUserKey(value: unknown): value is string {
  return typeof value === "string" && USER_KEY_PATTERN.test(value);
}

export function makeOwnerId(userKey: string): string {
  return createHash("sha256").update(userKey).digest("hex");
}

// ---------- Filenames ----------

// Multer reads filenames as "latin1", so a name like "café.txt" arrives garbled ("cafÃ©.txt").
// Converting the bytes back as UTF-8 repairs it. If the result looks wrong we keep the original.
export function fixFilenameEncoding(name: string): string {
  if (/[^\u0000-\u00FF]/.test(name)) {
    return name; // it already contains real non-Latin characters, so it was decoded correctly: leave it
  }
  const repaired = Buffer.from(name, "latin1").toString("utf8");
  return repaired.includes("�") ? name : repaired;
}

// Makes an uploaded filename safe: keeps only the last part of a path ("../../a.txt" -> "a.txt"),
// removes control characters, and limits the length. (We never use it as a path on disk.)
export function sanitizeFilename(originalName: string): string {
  const lastPart = originalName.split(/[\\/]/).pop() || "";
  const cleaned = lastPart.replace(/[\u0000-\u001F\u007F]/g, "").trim();
  if (cleaned === "" || cleaned === "." || cleaned === "..") {
    return "document";
  }
  return cleaned.slice(-200); // keep the end so the file extension survives
}
