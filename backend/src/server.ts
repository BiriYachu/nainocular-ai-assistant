// The web server: it only receives requests, calls the real work in documents.ts / chatbot.ts /
// database.ts, and sends JSON back. The list of URLs (endpoints) is in the middle of this file.
import "dotenv/config"; // loads backend/.env (the API keys stay on the backend)
import cors from "cors";
import express from "express";
import type { NextFunction, Request, RequestHandler, Response } from "express";
import rateLimit from "express-rate-limit";
import helmet from "helmet";
import multer from "multer";
import { answerQuestion } from "./chatbot";
import {
  closeDatabase,
  connectToDatabase,
  deleteConversation,
  findConversation,
  findDocument,
  getMessages,
  listConversations,
  listDocuments,
  markInterruptedDocuments,
  pingDatabase,
  publicDocument,
} from "./database";
import { removeDocument, uploadDocument } from "./documents";
import { AppError, isValidId, isValidUserKey, makeOwnerId, numberSetting } from "./helpers";
import {
  connectToVectorDatabase,
  deleteDocumentVectors,
  moveOldEmbeddings,
  pingVectorDatabase,
} from "./vectors";

const app = express();

// ---------- Security and basic setup ----------

// Behind a proxy (for example on a hosting service) every request seems to come from the proxy.
// TRUST_PROXY=1 tells Express to use the real visitor address, so the rate limits work per visitor.
if (numberSetting("TRUST_PROXY", 0) > 0) {
  app.set("trust proxy", numberSetting("TRUST_PROXY", 0));
}

app.use(helmet()); // safer HTTP headers (also hides "X-Powered-By: Express")
app.use(
  cors({
    origin: process.env.FRONTEND_ORIGIN || "http://localhost:5173", // only our frontend may call us
    allowedHeaders: ["Content-Type", "X-User-Key"],
  })
);
app.use(express.json({ limit: "10kb" })); // reads JSON bodies, and stops huge ones

// Answers contain private data, so browsers and proxies must never keep a copy.
app.use("/api", (_req, res, next) => {
  res.set("Cache-Control", "no-store");
  next();
});

// Upload and chat call paid AI services, so they get a limit per IP address per minute.
// (The limit is a function, so the number is read from .env on every request.)
const tooManyRequests = { success: false, message: "Too many requests. Please wait a moment and try again." };
const uploadLimiter = rateLimit({
  windowMs: 60 * 1000, limit: () => numberSetting("RATE_LIMIT_UPLOAD_PER_MINUTE", 10), message: tooManyRequests,
});
const chatLimiter = rateLimit({
  windowMs: 60 * 1000, limit: () => numberSetting("RATE_LIMIT_CHAT_PER_MINUTE", 20), message: tooManyRequests,
});

// Multer reads uploaded files. memoryStorage keeps the file in memory, so nothing is written to disk.
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_SIZE, files: 1, fields: 5, parts: 10 },
});

// Express 4 does not catch errors thrown in async functions, so this passes them to the error handler.
function safely(handler: (req: Request, res: Response) => Promise<void>): RequestHandler {
  return (req, res, next) => {
    handler(req, res).catch(next);
  };
}

// ---------- Who is asking? ----------

// Every request for data must carry the user's private key in the "X-User-Key" header.
// We turn it into an "owner ID" (a hash) and remember it for this request.
function identifyUser(req: Request, res: Response, next: NextFunction) {
  const userKey = req.header("x-user-key");
  if (!isValidUserKey(userKey)) {
    next(new AppError("Missing or invalid user key.", 401));
    return;
  }
  res.locals.ownerId = makeOwnerId(userKey);
  next();
}

// Reads the owner ID that identifyUser saved for this request.
function ownerOf(res: Response): string {
  return res.locals.ownerId;
}

// Reads an ID from the URL and rejects anything that is not a valid ID.
function readIdFromUrl(value: unknown, name: string): string {
  if (!isValidId(value)) {
    throw new AppError("Invalid " + name + ".", 400);
  }
  return value;
}

// ---------- Endpoints ----------

// Is the server running, with MongoDB and Qdrant? (No user key needed.)
app.get("/api/health", safely(async (_req, res) => {
  const databaseIsUp = await pingDatabase();
  const vectorsAreUp = await pingVectorDatabase();
  if (!databaseIsUp || !vectorsAreUp) {
    const missing = [databaseIsUp ? "" : "MongoDB", vectorsAreUp ? "" : "Qdrant (the vector database)"].filter((name) => name !== "");
    res.status(503).json({ success: false, status: "degraded", message: "Server is running, but this is not available: " + missing.join(" and ") + "." });
    return;
  }
  res.json({ success: true, status: "ok", message: "Server is running", database: "connected", vectorDatabase: "connected" });
}));

// All the routes below need a user key.
app.use(["/api/documents", "/api/chat", "/api/conversations"], identifyUser);

// The list of this user's documents (newest first).
app.get("/api/documents", safely(async (_req, res) => {
  const documents = await listDocuments(ownerOf(res));
  res.json({ success: true, documents: documents.map(publicDocument) });
}));

// Upload a PDF or TXT file. The form field must be called "file".
app.post("/api/documents/upload", uploadLimiter, upload.single("file"), safely(async (req, res) => {
  if (!req.file) {
    throw new AppError('No file was uploaded. Send a file in the form field named "file".', 400);
  }
  const result = await uploadDocument(ownerOf(res), req.file);
  res.json({
    success: true,
    message: result.alreadyProcessed ? "This file was already processed. Reusing the saved data." : "Document processed successfully",
    document: result.document,
  });
}));

// Read one saved document.
app.get("/api/documents/:documentId", safely(async (req, res) => {
  const document = await findDocument(ownerOf(res), readIdFromUrl(req.params.documentId, "document ID"));
  if (document === null) {
    throw new AppError("Document not found.", 404);
  }
  res.json({ success: true, document: publicDocument(document) });
}));

// Delete a document with its chunks, vectors and chats.
app.delete("/api/documents/:documentId", safely(async (req, res) => {
  const result = await removeDocument(ownerOf(res), readIdFromUrl(req.params.documentId, "document ID"));
  if (result === "not-found") {
    throw new AppError("Document not found.", 404);
  }
  if (result === "busy") {
    throw new AppError("This document is still being processed. Please try again in a moment.", 409);
  }
  res.json({ success: true, message: "Document removed." });
}));

// Ask a question about one or more documents. Body: { question, documentIds: [...], conversationId? }
app.post("/api/chat", chatLimiter, safely(async (req, res) => {
  res.json(await answerQuestion(ownerOf(res), req.body ?? {}));
}));

// This user's saved chats, newest first.
app.get("/api/conversations", safely(async (_req, res) => {
  const conversations = await listConversations(ownerOf(res));
  res.json({
    success: true,
    conversations: conversations.map((chat) => ({
      id: chat.id,
      title: chat.title,
      documentIds: chat.documentIds,
      createdAt: chat.createdAt.toISOString(),
      updatedAt: chat.updatedAt.toISOString(),
    })),
  });
}));

// The messages of one saved chat, oldest first.
app.get("/api/conversations/:conversationId/messages", safely(async (req, res) => {
  const conversationId = readIdFromUrl(req.params.conversationId, "conversation ID");
  if ((await findConversation(ownerOf(res), conversationId)) === null) {
    throw new AppError("Conversation not found.", 404);
  }
  const messages = await getMessages(conversationId);
  res.json({
    success: true,
    conversationId: conversationId,
    messages: messages.map((message) => ({
      id: message.id,
      role: message.role,
      text: message.text,
      status: message.status,
      sources: message.sources,
      note: message.note,
      createdAt: message.createdAt.toISOString(),
    })),
  });
}));

// Delete one saved chat.
app.delete("/api/conversations/:conversationId", safely(async (req, res) => {
  const wasDeleted = await deleteConversation(ownerOf(res), readIdFromUrl(req.params.conversationId, "conversation ID"));
  if (!wasDeleted) {
    throw new AppError("Conversation not found.", 404);
  }
  res.json({ success: true, message: "Conversation removed." });
}));

// ---------- Errors ----------

// Unknown URLs.
app.use((_req, res) => {
  res.status(404).json({ success: false, message: "This route does not exist." });
});

// All errors end up here (Express recognises an error handler by its 4 parameters).
// Only short, safe messages go to the browser: never stack traces, file paths or secrets.
app.use((error: unknown, _req: Request, res: Response, _next: NextFunction) => {
  if (error instanceof AppError) {
    res.status(error.statusCode).json({ success: false, message: error.message }); // we wrote this message
  } else if (error instanceof multer.MulterError) {
    if (error.code === "LIMIT_FILE_SIZE") {
      res.status(413).json({ success: false, message: "The file is larger than 10 MB." });
    } else if (error.code === "LIMIT_UNEXPECTED_FILE") {
      res.status(400).json({ success: false, message: 'Upload exactly one file in the form field named "file".' });
    } else {
      res.status(400).json({ success: false, message: "The upload request was invalid." });
    }
  } else if ((error as { type?: string } | null)?.type === "entity.too.large") {
    res.status(413).json({ success: false, message: "The request body is too large." });
  } else if (error instanceof SyntaxError) {
    res.status(400).json({ success: false, message: "The request body is not valid JSON." });
  } else {
    // Unexpected: log only the name and message for us, and show a general message to the user.
    console.error("Unexpected error:", error instanceof Error ? error.name + ": " + error.message : String(error));
    res.status(500).json({ success: false, message: "Something went wrong on the server." });
  }
});

export default app; // the tests import the app without starting the server

// ---------- Start ----------

// Runs only when this file is started directly (npm run dev), not when the tests import it.
if (require.main === module) {
  const port = Number(process.env.PORT) || 5000;

  // A promise that nobody handled is a bug. Log it (briefly) instead of silently ignoring it.
  process.on("unhandledRejection", (reason) => {
    console.error("Unhandled promise rejection:", reason instanceof Error ? reason.name + ": " + reason.message : String(reason));
  });

  // Start-up steps, one after the other. If one fails, we stop with a clear (and safe) message.
  async function prepare() {
    try {
      await connectToDatabase();
    } catch {
      // Do not print the error itself: it could contain the connection string.
      throw new Error("Could not start: connecting to MongoDB failed. Check MONGODB_URI in backend/.env and that MongoDB is running.");
    }
    try {
      await connectToVectorDatabase();
    } catch (error) {
      if (error instanceof AppError) {
        throw new Error("Could not start: " + error.message); // for example "the vector size does not match"
      }
      throw new Error("Could not start: connecting to Qdrant (the vector database) failed. Check QDRANT_URL in backend/.env and that Qdrant is running.");
    }

    // An older version stored embeddings inside MongoDB. Move them to Qdrant (does nothing when there are none).
    const movedCount = await moveOldEmbeddings();
    if (movedCount > 0) {
      console.log("Moved " + movedCount + " embedding(s) from MongoDB to Qdrant.");
    }

    // If the server stopped while a file was being processed, that file would stay "processing" forever.
    const interruptedIds = await markInterruptedDocuments();
    for (const documentId of interruptedIds) {
      await deleteDocumentVectors(documentId).catch(() => {});
    }
    if (interruptedIds.length > 0) {
      console.log(interruptedIds.length + " unfinished upload(s) from the last run were marked as failed.");
    }
  }

  prepare()
    .then(async () => {
      const server = app.listen(port, () => console.log("Backend running at http://localhost:" + port));

      // Ctrl+C (or a restart): stop taking new requests, finish the current ones, close the database.
      const shutDown = () => {
        console.log("Shutting down...");
        server.close(() => {
          closeDatabase().finally(() => process.exit(0));
        });
        setTimeout(() => process.exit(1), 5000).unref(); // if something hangs, leave after 5 seconds
      };
      process.on("SIGINT", shutDown);
      process.on("SIGTERM", shutDown);
    })
    .catch((error) => {
      console.error(error instanceof Error ? error.message : "Could not start the backend.");
      process.exit(1);
    });
}
