// Everything that talks to Qdrant (the vector database) lives in this file.
//
// MongoDB keeps the TEXT and the details of every chunk. Qdrant keeps the EMBEDDING (the list of
// 1536 numbers) of every chunk, and it finds the chunks whose meaning is closest to a question.
//
// One "point" in Qdrant = one chunk. A point has:
//   - an id: the SAME id the chunk has in MongoDB (so we can fetch the text afterwards)
//   - a vector: the embedding
//   - a small payload: the ownerId and documentId, used to FILTER every search, so a search only
//     looks at the chosen documents of one user
import { QdrantClient } from "@qdrant/js-client-rest";
import {
  findOwnerIds,
  getChunksWithOldEmbeddings,
  removeEmbeddingsFromChunks,
} from "./database";
import { AppError, numberSetting } from "./helpers";

let client: QdrantClient | null = null;
let collectionName = "";
let vectorSize = 0;

// The size of every embedding. It must match the embedding model (EMBEDDING_DIMENSIONS in .env).
function readVectorSize(): number {
  const size = numberSetting("EMBEDDING_DIMENSIONS", 1536);
  return size > 0 ? size : 1536;
}

// ---------- Connecting ----------

// Connects to Qdrant and makes sure the collection (think: a table of points) exists.
// "Cosine" tells Qdrant to compare vectors by cosine similarity: (a . b) / (|a| x |b|).
export async function connectToVectorDatabase(url?: string, name?: string): Promise<void> {
  client = new QdrantClient({
    url: url || process.env.QDRANT_URL || "http://localhost:6333",
    apiKey: process.env.QDRANT_API_KEY || undefined,
    checkCompatibility: false,
  });
  collectionName = name || process.env.QDRANT_COLLECTION || "nainocular_chunks";
  vectorSize = readVectorSize();

  const existing = await client.getCollections();
  const alreadyThere = existing.collections.some((collection) => collection.name === collectionName);

  if (alreadyThere) {
    // The collection was made for a certain embedding size. If the model (and its size) changed, stop clearly.
    const info = await client.getCollection(collectionName);
    const vectorsConfig = info.config.params.vectors as { size?: number } | undefined;
    if (vectorsConfig?.size !== vectorSize) {
      throw new AppError(
        "The Qdrant collection stores vectors of size " + vectorsConfig?.size + " but EMBEDDING_DIMENSIONS is " +
          vectorSize + ". Use another QDRANT_COLLECTION name, or delete the old collection.",
        500
      );
    }
  } else {
    await client.createCollection(collectionName, { vectors: { size: vectorSize, distance: "Cosine" } });
  }

  // Indexes on the payload fields make the filters fast.
  await client.createPayloadIndex(collectionName, { field_name: "ownerId", field_schema: "keyword", wait: true });
  await client.createPayloadIndex(collectionName, { field_name: "documentId", field_schema: "keyword", wait: true });
}

export function closeVectorDatabase(): void {
  client = null;
}

// Used by the health check: "is Qdrant there?"
export async function pingVectorDatabase(): Promise<boolean> {
  try {
    await vectorClient().getCollections();
    return true;
  } catch {
    return false;
  }
}

// Used by tests to remove their temporary collection.
export async function dropVectorCollection(): Promise<void> {
  if (client !== null) {
    await client.deleteCollection(collectionName);
  }
}

// The Qdrant client. Throws a clear error if we are not connected.
function vectorClient(): QdrantClient {
  if (client === null) {
    throw new AppError("The vector database is not available. Please try again later.", 503);
  }
  return client;
}

// Turns a Qdrant problem into a short, safe message for the user (details are only logged, briefly).
function toAppError(error: unknown): AppError {
  if (error instanceof AppError) {
    return error;
  }
  const message = error instanceof Error ? error.message : String(error);
  const data = (error as { data?: unknown } | null)?.data;
  const details = message + " " + (data ? JSON.stringify(data) : "");

  if (/dimension/i.test(details)) {
    return new AppError(
      "The stored embeddings do not match the current embedding size (EMBEDDING_DIMENSIONS). Please upload the documents again.",
      409
    );
  }
  if (/fetch failed|ECONNREFUSED|ECONNRESET|timed out|timeout/i.test(details)) {
    return new AppError("The vector database is not available. Please try again later.", 503);
  }
  console.error("Vector database error:", message.slice(0, 200));
  return new AppError("The vector database returned an error.", 502);
}

// ---------- Saving, searching, deleting ----------

// One embedding to store: which chunk it belongs to, whose it is, and the numbers.
export type VectorToSave = { id: string; ownerId: string; documentId: string; embedding: number[] };

const SAVE_BATCH_SIZE = 100;

// Stores embeddings in Qdrant. If a point with the same id exists, it is replaced (never duplicated).
export async function saveVectors(vectors: VectorToSave[]): Promise<void> {
  try {
    for (let start = 0; start < vectors.length; start += SAVE_BATCH_SIZE) {
      const batch = vectors.slice(start, start + SAVE_BATCH_SIZE);
      await vectorClient().upsert(collectionName, {
        wait: true, // return only when the points are really stored
        points: batch.map((item) => ({
          id: item.id,
          vector: item.embedding,
          payload: { ownerId: item.ownerId, documentId: item.documentId },
        })),
      });
    }
  } catch (error) {
    throw toAppError(error);
  }
}

// A search result: which chunk, and its cosine similarity to the question (1 = same meaning, 0 = unrelated).
export type VectorHit = { chunkId: string; similarity: number };

// Finds the chunks closest to the question, best first. The filter makes sure Qdrant only looks at the
// chosen documents of THIS user. (Qdrant returns the ids and scores; the text comes from MongoDB.)
export async function searchVectors(
  ownerId: string,
  documentIds: string[],
  questionVector: number[],
  limit: number
): Promise<VectorHit[]> {
  try {
    const result = await vectorClient().query(collectionName, {
      query: questionVector,
      limit: limit,
      filter: {
        must: [
          { key: "ownerId", match: { value: ownerId } },
          { key: "documentId", match: { any: documentIds } },
        ],
      },
      with_payload: false,
    });
    return result.points.map((point) => ({ chunkId: String(point.id), similarity: point.score }));
  } catch (error) {
    throw toAppError(error);
  }
}

// How many vectors Qdrant holds for a document. Used to notice if the search index lost data.
export async function countDocumentVectors(documentId: string): Promise<number> {
  try {
    const result = await vectorClient().count(collectionName, {
      filter: { must: [{ key: "documentId", match: { value: documentId } }] },
      exact: true,
    });
    return result.count;
  } catch (error) {
    throw toAppError(error);
  }
}

// How many vectors Qdrant holds in total. Used by tests, to check that nothing is left behind.
export async function countAllVectors(): Promise<number> {
  try {
    return (await vectorClient().count(collectionName, { exact: true })).count;
  } catch (error) {
    throw toAppError(error);
  }
}

// Removes all vectors of a document.
export async function deleteDocumentVectors(documentId: string): Promise<void> {
  try {
    await vectorClient().delete(collectionName, {
      wait: true,
      filter: { must: [{ key: "documentId", match: { value: documentId } }] },
    });
  } catch (error) {
    throw toAppError(error);
  }
}

// ---------- Moving old embeddings out of MongoDB ----------

// An older version of this app stored the embeddings inside the MongoDB chunks. At startup we copy any such
// embeddings into Qdrant and remove them from MongoDB, so existing documents keep working.
// Embeddings that cannot be moved (wrong size, or their document no longer exists) are left untouched.
// Returns how many were moved. Running it again is safe: it only touches chunks that still have an embedding.
export async function moveOldEmbeddings(): Promise<number> {
  let moved = 0;
  let lastId = "";

  while (true) {
    const oldChunks = await getChunksWithOldEmbeddings(lastId, 100);
    if (oldChunks.length === 0) {
      break;
    }
    lastId = oldChunks[oldChunks.length - 1].id;

    const owners = await findOwnerIds(oldChunks.map((chunk) => chunk.documentId));
    const readyToMove = oldChunks.filter((chunk) => {
      const embedding = chunk.embedding;
      const isUsable = Array.isArray(embedding) && embedding.length === vectorSize && embedding.every((n) => Number.isFinite(n));
      return isUsable && owners.has(chunk.documentId);
    });

    await saveVectors(
      readyToMove.map((chunk) => ({
        id: chunk.id,
        ownerId: owners.get(chunk.documentId) as string,
        documentId: chunk.documentId,
        embedding: chunk.embedding as number[], // safe: the filter above kept only chunks with a usable embedding
      }))
    );
    await removeEmbeddingsFromChunks(readyToMove.map((chunk) => chunk.id));
    moved += readyToMove.length;
  }
  return moved;
}
