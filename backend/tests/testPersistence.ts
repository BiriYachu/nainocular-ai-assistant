// Proves that data survives a BACKEND RESTART.
// It starts the real backend (dist/server.js) as a separate process, uploads and chats,
// KILLS the process, starts a brand new one, and checks that everything is still there.
//
// Uses the LOCAL MOCK providers (no real AI) and a temporary MongoDB database.
// Needs MongoDB running locally and the backend built first:  npm run build
// Run with: npm run test:persistence
import { ChildProcess, spawn } from "child_process";
import fs from "fs";
import path from "path";
import { createHash, randomUUID } from "crypto";
import { MongoClient } from "mongodb";
import { startMockProviders } from "./mockProviders";

let failures = 0;
function check(name: string, condition: boolean, detail?: unknown) {
  console.log((condition ? "PASS " : "FAIL ") + name);
  if (!condition) {
    failures++;
    if (detail !== undefined) console.log("     got:", JSON.stringify(detail));
  }
}

const BIOLOGY_TEXT =
  "DNA replication is the process by which a cell copies its DNA. " +
  "The enzyme helicase unwinds the double helix at the replication fork. " +
  "DNA polymerase adds new nucleotides to the growing strand.";

const PORT = 5071;
const BASE = "http://localhost:" + PORT;

function startBackend(env: NodeJS.ProcessEnv): Promise<ChildProcess> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["dist/server.js"], { env: { ...process.env, ...env }, stdio: ["ignore", "pipe", "pipe"] });
    const timer = setTimeout(() => reject(new Error("Backend did not start in time")), 20000);
    child.stdout?.on("data", (data) => {
      if (String(data).includes("Backend running")) {
        clearTimeout(timer);
        resolve(child);
      }
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error("Backend exited early with code " + code));
    });
  });
}

function stopBackend(child: ChildProcess): Promise<void> {
  return new Promise((resolve) => {
    child.removeAllListeners("exit");
    child.on("exit", () => resolve());
    child.kill();
  });
}

// Two different users (each browser has its own private key of 64 hex characters).
const KEY = "d4".repeat(32);
const OTHER_KEY = "e5".repeat(32);
const OWNER_ID = createHash("sha256").update(KEY).digest("hex"); // how the backend names this user in MongoDB

async function api(method: string, urlPath: string, body?: unknown, key: string = KEY) {
  const headers: Record<string, string> = { "X-User-Key": key };
  if (body) headers["Content-Type"] = "application/json";
  const response = await fetch(BASE + urlPath, { method, headers, body: body ? JSON.stringify(body) : undefined });
  return { status: response.status, body: await response.json() };
}

async function upload(name: string, text: string) {
  const form = new FormData();
  form.append("file", new Blob([text], { type: "text/plain" }), name);
  const response = await fetch(BASE + "/api/documents/upload", { method: "POST", headers: { "X-User-Key": KEY }, body: form });
  return { status: response.status, body: await response.json() };
}

async function main() {
  if (!fs.existsSync(path.join(process.cwd(), "dist", "server.js"))) {
    console.error("dist/server.js not found. Run `npm run build` first.");
    process.exit(1);
  }

  const mock = await startMockProviders(0);
  const databaseName = "nainocular_test_persist_" + Date.now();
  const mongoUri = process.env.MONGODB_URI || "mongodb://localhost:27017";

  const childEnv: NodeJS.ProcessEnv = {
    PORT: String(PORT),
    MONGODB_URI: mongoUri,
    MONGODB_DATABASE: databaseName,
    EMBEDDING_API_URL: mock.baseUrl + "/embeddings",
    OPENROUTER_API_URL: mock.baseUrl + "/chat",
    OPENROUTER_API_KEY: "test-key",
    EMBEDDING_API_KEY: "",
    OPENROUTER_MODEL: "mock-model",
    EMBEDDING_DIMENSIONS: "1536",
    RETRIEVAL_MIN_SIMILARITY: "0.1",
    ENABLE_CLAIM_VERIFICATION: "true",
    QDRANT_COLLECTION: databaseName, // a temporary Qdrant collection, deleted at the end
  };

  const mongo = new MongoClient(mongoUri);
  await mongo.connect();
  const testDb = mongo.db(databaseName);

  // The test also talks to Qdrant directly, to look at what the backend stored there.
  process.env.QDRANT_COLLECTION = databaseName;
  process.env.EMBEDDING_DIMENSIONS = "1536";
  const vectors = await import("../src/vectors");

  console.log("\n--- Backend process #1 ---");
  let backend = await startBackend(childEnv);
  await vectors.connectToVectorDatabase(); // (the backend has created the collection by now)
  const uploaded = await upload("biology.txt", BIOLOGY_TEXT);
  check("upload works", uploaded.status === 200 && uploaded.body.document.status === "processed", uploaded.body);
  const documentId: string = uploaded.body.document.id;
  const chunkIdBefore: string = (await testDb.collection("chunks").findOne({ documentId }))?.id;
  check("the embeddings are in Qdrant (one per chunk)", (await vectors.countDocumentVectors(documentId)) === uploaded.body.document.chunkCount);
  const firstAnswer = await api("POST", "/api/chat", { question: "What does helicase do?", documentIds: [documentId] });
  check("question answered before restart", firstAnswer.body.found === true, firstAnswer.body);
  const conversationId: string = firstAnswer.body.conversationId;

  // Simulate a crash in the middle of an upload: a document that is stuck in "processing".
  const stuckId = "3f2b8c1e-bbbb-4abc-9def-0123456789ab";
  await testDb.collection("documents").insertOne({
    id: stuckId, ownerId: OWNER_ID, filename: "stuck.txt", fileType: "txt", fingerprint: "stuck", status: "processing",
    characterCount: 0, chunkCount: 0, uploadedAt: new Date(),
  });

  // An OLD-STYLE chunk: its embedding is stored inside MongoDB, like an older version of the app did.
  const oldDocumentId = "3f2b8c1e-cccc-4abc-9def-0123456789ab";
  const oldChunkId = randomUUID();
  const oldVector = new Array(1536).fill(0);
  oldVector[0] = 1;
  await testDb.collection("documents").insertOne({
    id: oldDocumentId, ownerId: OWNER_ID, filename: "old-style.txt", fileType: "txt", fingerprint: "old-style", status: "processed",
    characterCount: 5, chunkCount: 1, uploadedAt: new Date(),
  });
  await testDb.collection("chunks").insertOne({ id: oldChunkId, documentId: oldDocumentId, chunkIndex: 0, text: "old text", filename: "old-style.txt", embedding: oldVector });

  console.log("\n--- KILLING the backend process, starting a new one ---");
  await stopBackend(backend);
  let stillRunning = true;
  try {
    await fetch(BASE + "/api/health", { signal: AbortSignal.timeout(1500) });
  } catch {
    stillRunning = false;
  }
  check("old backend is really gone", stillRunning === false);
  backend = await startBackend(childEnv);

  console.log("\n--- Backend process #2 (fresh memory) ---");
  const document = await api("GET", "/api/documents/" + documentId);
  check("same document is still there", document.status === 200 && document.body.document.filename === "biology.txt" && document.body.document.status === "processed", document.body);
  check("document metadata preserved (counts, timestamp)",
    document.body.document.chunkCount === uploaded.body.document.chunkCount &&
    document.body.document.characterCount === uploaded.body.document.characterCount &&
    document.body.document.uploadedAt === uploaded.body.document.uploadedAt);

  const afterRestart = await api("POST", "/api/chat", { question: "What does helicase do?", documentIds: [documentId], conversationId });
  check("question about the OLD document works after restart (no re-upload)", afterRestart.status === 200 && afterRestart.body.found === true, afterRestart.body);
  check("answer cites the same chunk ID as before the restart", afterRestart.body.sources?.[0]?.chunkId === chunkIdBefore);
  check("the chat continues in the same conversation", afterRestart.body.conversationId === conversationId);
  const unrelated = await api("POST", "/api/chat", { question: "What is the capital of Australia?", documentIds: [documentId] });
  check("unrelated question still gives the exact fallback", unrelated.body.found === false && unrelated.body.answer === "I couldn't find this information in the uploaded documents.", unrelated.body);

  const history = await api("GET", "/api/conversations/" + conversationId + "/messages");
  check("chat history from BEFORE the restart is still there (2 old + 2 new messages)",
    history.status === 200 && history.body.messages.length === 4, history.body);
  check("old answer and its citation are preserved",
    history.body.messages[1].text === firstAnswer.body.answer && history.body.messages[1].sources[0].chunkId === chunkIdBefore);
  const conversations = await api("GET", "/api/conversations");
  check("the user's chats can be listed after the restart, with their title",
    conversations.body.conversations.some((c: { id: string; title: string }) => c.id === conversationId && c.title === "What does helicase do?"), conversations.body);

  const library = await api("GET", "/api/documents");
  check("the user's document library is still there (no re-upload needed)", library.body.documents.some((d: { id: string }) => d.id === documentId), library.body);
  check("another user sees none of it",
    (await api("GET", "/api/documents", undefined, OTHER_KEY)).body.documents.length === 0 &&
    (await api("GET", "/api/documents/" + documentId, undefined, OTHER_KEY)).status === 404 &&
    (await api("GET", "/api/conversations/" + conversationId + "/messages", undefined, OTHER_KEY)).status === 404);
  const movedChunk = await testDb.collection("chunks").findOne({ id: oldChunkId });
  check("at startup, an old embedding was moved out of MongoDB", movedChunk !== null && movedChunk.embedding === undefined, movedChunk);
  check("...into Qdrant, owned by the same user", (await vectors.searchVectors(OWNER_ID, [oldDocumentId], oldVector, 3)).some((hit) => hit.chunkId === oldChunkId));
  check("the vectors of the normal document are still in Qdrant after the restart", (await vectors.countDocumentVectors(documentId)) === uploaded.body.document.chunkCount);
  const recoveredStuck = await api("GET", "/api/documents/" + stuckId);
  check("an upload that was interrupted by the restart is marked failed",
    recoveredStuck.body.document?.status === "failed" && /interrupted/i.test(recoveredStuck.body.document?.errorMessage ?? ""), recoveredStuck.body);

  console.log("\n--- Reprocessing without duplicates ---");
  const reuploaded = await upload("biology.txt", BIOLOGY_TEXT);
  check("same file uploaded again returns the SAME document ID", reuploaded.body.document.id === documentId, reuploaded.body);
  check("only one document record exists", (await testDb.collection("documents").countDocuments({ filename: "biology.txt" })) === 1);
  check("chunk count in MongoDB unchanged (no duplicates)",
    (await testDb.collection("chunks").countDocuments({ documentId })) === uploaded.body.document.chunkCount);
  const rawChunk = await testDb.collection("chunks").findOne({ documentId });
  check("MongoDB holds the chunk text but no embedding", typeof rawChunk?.text === "string" && rawChunk?.embedding === undefined);
  check("no duplicate vectors in Qdrant after uploading the same file again", (await vectors.countDocumentVectors(documentId)) === uploaded.body.document.chunkCount);

  console.log("\n--- Backend cannot reach MongoDB ---");
  const badStart = await new Promise<{ code: number | null; stderr: string }>((resolve) => {
    const child = spawn(process.execPath, ["dist/server.js"], {
      env: { ...process.env, ...childEnv, PORT: String(PORT + 1), MONGODB_URI: "mongodb://localhost:1/secret-db-name" },
      stdio: ["ignore", "ignore", "pipe"],
    });
    let stderr = "";
    child.stderr?.on("data", (data) => (stderr += data));
    child.on("exit", (code) => resolve({ code, stderr }));
  });
  check("backend refuses to start without MongoDB (exit code 1)", badStart.code === 1, badStart.code);
  check("it prints a clear message", /connecting to MongoDB failed/.test(badStart.stderr), badStart.stderr);
  check("the message does not leak the connection string", !badStart.stderr.includes("secret-db-name") && !badStart.stderr.includes("localhost:1"));
  console.log("\n--- Backend cannot reach Qdrant ---");
  const noQdrant = await new Promise<{ code: number | null; stderr: string }>((resolve) => {
    const child = spawn(process.execPath, ["dist/server.js"], {
      env: { ...process.env, ...childEnv, PORT: String(PORT + 1), QDRANT_URL: "http://localhost:1/secret-path" },
      stdio: ["ignore", "ignore", "pipe"],
    });
    let stderr = "";
    child.stderr?.on("data", (data) => (stderr += data));
    child.on("exit", (code) => resolve({ code, stderr }));
  });
  check("backend refuses to start without Qdrant (exit code 1)", noQdrant.code === 1, noQdrant.code);
  check("it prints a clear message", /Qdrant/.test(noQdrant.stderr), noQdrant.stderr);
  check("the message does not leak the address", !noQdrant.stderr.includes("secret-path") && !noQdrant.stderr.includes("localhost:1"));

  console.log("\n--- Cleanup ---");
  await vectors.dropVectorCollection();
  await stopBackend(backend);
  mock.server.close();
  await testDb.dropDatabase();
  await mongo.close();

  console.log(failures === 0 ? "\nAll persistence checks passed." : "\n" + failures + " check(s) FAILED.");
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
