// Everything about talking to the backend: the types of the data, the private user key,
// and one small function for each thing the app can ask the backend to do.

// ======================================================================
// 1. TYPES: the shape of the data (so TypeScript can warn us about mistakes)
// ======================================================================

// A document as the backend describes it.
export type UploadedDocument = {
  id: string;
  filename: string;
  fileType: "pdf" | "txt";
  status: "processing" | "processed" | "failed";
  characterCount: number;
  chunkCount: number;
  uploadedAt: string;
  warning?: string; // for example "1 of 8 pages had no readable text"
  errorMessage?: string; // why a document failed
};

// A piece of evidence shown under an answer.
export type Source = {
  filename: string;
  excerpt: string;
  pageNumber?: number; // only PDFs have page numbers
};

// One chat message. User messages have no sources.
export type Message = {
  role: "user" | "assistant";
  text: string;
  note?: string; // a small remark, for example "could not be verified"
  sources?: Source[];
};

// One saved chat, as shown in the chat list.
export type ChatSummary = {
  id: string;
  title: string; // the first question
  documentIds: string[]; // the documents this chat is about
  updatedAt: string;
};

// ======================================================================
// 2. THE PRIVATE USER KEY
// ======================================================================
// There are no passwords. This browser makes a random secret key the first time it opens the app and
// keeps it in localStorage. It is sent with every request, and the backend uses it to know whose
// documents and chats to show. Another browser (or cleared browser data) means a new, empty library.

const KEY_NAME = "nainocular-user-key";
let userKey = ""; // the key for this visit

// Makes 32 random bytes and writes them as 64 hexadecimal characters.
function createKey(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function getUserKey(): string {
  if (userKey !== "") {
    return userKey;
  }
  try {
    const saved = localStorage.getItem(KEY_NAME);
    if (saved !== null && /^[0-9a-f]{64}$/i.test(saved)) {
      userKey = saved;
      return userKey;
    }
  } catch {
    // Storage is not available (for example private browsing). We use a key for this visit only.
  }
  userKey = createKey();
  try {
    localStorage.setItem(KEY_NAME, userKey);
  } catch {
    // Not saved: the library will be empty next time. That is the best we can do here.
  }
  return userKey;
}

// ======================================================================
// 3. TALKING TO THE BACKEND
// ======================================================================

// The address of the backend. It can be changed with a VITE_API_URL setting.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
const REQUEST_TIMEOUT_MS = 120000; // give up after 2 minutes

type RequestOptions = { method: string; body?: BodyInit; headers?: Record<string, string> };

// Sends a request and returns the JSON answer. Throws an Error with a readable message if it fails.
async function callBackend<T>(path: string, options: RequestOptions): Promise<T> {
  let response: Response;
  try {
    response = await fetch(API_URL + path, {
      method: options.method,
      body: options.body,
      headers: { ...options.headers, "X-User-Key": getUserKey() },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new Error("The server took too long to answer. Please try again.");
    }
    // fetch only throws when no request could be made at all (backend off, no network)
    throw new Error("Could not reach the server. Make sure the backend is running.");
  }

  let data: { message?: string } | null = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    throw new Error(data?.message || "The server could not process this request.");
  }
  return data as T;
}

export async function listDocuments(): Promise<UploadedDocument[]> {
  const result = await callBackend<{ documents: UploadedDocument[] }>("/api/documents", { method: "GET" });
  return result.documents;
}

// Sends a file to the backend. It extracts the text, makes chunks and embeddings, and saves them.
export async function uploadFile(file: File): Promise<UploadedDocument> {
  const formData = new FormData();
  formData.append("file", file); // "file" must match the field name the backend expects

  // We do NOT set the Content-Type header ourselves: the browser adds the correct one for FormData.
  const result = await callBackend<{ document: UploadedDocument }>("/api/documents/upload", {
    method: "POST",
    body: formData,
  });
  return result.document;
}

export async function deleteDocument(documentId: string): Promise<void> {
  await callBackend("/api/documents/" + encodeURIComponent(documentId), { method: "DELETE" });
}

// Asks a question about one or more documents. conversationId is null for the first question of a chat,
// so the backend starts a new saved chat.
export async function askBackend(question: string, documentIds: string[], conversationId: string | null) {
  return callBackend<{ conversationId: string; answer: string; note?: string; sources: Source[] }>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" }, // here the body is JSON, so we say so
    body: JSON.stringify({ question, documentIds, conversationId: conversationId ?? undefined }),
  });
}

export async function listChats(): Promise<ChatSummary[]> {
  const result = await callBackend<{ conversations: ChatSummary[] }>("/api/conversations", { method: "GET" });
  return result.conversations;
}

export async function loadChatMessages(chatId: string): Promise<Message[]> {
  const result = await callBackend<{ messages: Message[] }>(
    "/api/conversations/" + encodeURIComponent(chatId) + "/messages",
    { method: "GET" }
  );
  return result.messages;
}

export async function deleteChat(chatId: string): Promise<void> {
  await callBackend("/api/conversations/" + encodeURIComponent(chatId), { method: "DELETE" });
}
