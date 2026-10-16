// Talking to the AI provider (OpenRouter): one function for embeddings, one for chat.
// API keys are read from backend/.env here and never leave the backend.
import { AppError, numberSetting } from "./helpers";

const REQUEST_TIMEOUT_MS = 60000;
const EMBEDDING_BATCH_SIZE = 64; // how many texts we send to the provider in one request

const MAX_ATTEMPTS = 3; // how many times we try a provider request before giving up

// Calls the provider. If the network fails, or the provider says "too many requests" (429) or has a
// temporary server problem (500 or higher), we wait a little and try again (up to 3 tries in total).
// Other answers (for example 401 "wrong key") are returned at once, because trying again cannot help.
async function fetchWithRetry(url: string, options: RequestInit): Promise<Response> {
  const delay = numberSetting("PROVIDER_RETRY_DELAY_MS", 1000);

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const isLastAttempt = attempt === MAX_ATTEMPTS;
    try {
      const response = await fetch(url, { ...options, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
      const isTemporaryProblem = response.status === 429 || response.status >= 500;
      if (!isTemporaryProblem || isLastAttempt) {
        return response;
      }
    } catch (error) {
      if (isLastAttempt) {
        throw error; // the caller turns this into a clear "could not reach the provider" message
      }
    }
    await new Promise((resolve) => setTimeout(resolve, delay * attempt)); // wait longer each time
  }
  throw new Error("unreachable");
}
// Pulls a short, readable message out of a provider error response.
function readProviderMessage(responseText: string): string {
  try {
    const data = JSON.parse(responseText);
    const message = data?.error?.message || data?.message;
    if (typeof message === "string") {
      return message.slice(0, 200);
    }
  } catch {
    // not JSON: fall through
  }
  return "no details provided";
}

// ---------- Embeddings ----------

// An embedding is a list of numbers describing the meaning of a text.
// This sends ONE batch of texts and checks that the answer looks right.
async function requestEmbeddings(texts: string[]): Promise<number[][]> {
  const apiKey = process.env.EMBEDDING_API_KEY || process.env.OPENROUTER_API_KEY || "";
  if (apiKey === "") {
    throw new AppError(
      "Embedding API key is not configured. Set EMBEDDING_API_KEY (or OPENROUTER_API_KEY) in backend/.env.",
      503
    );
  }

  const expectedLength = numberSetting("EMBEDDING_DIMENSIONS", 1536); // 0 = do not check
  const body: Record<string, unknown> = {
    model: process.env.EMBEDDING_MODEL || "openai/text-embedding-3-small",
    input: texts,
    encoding_format: "float",
  };
  if (expectedLength > 0) {
    body.dimensions = expectedLength;
  }

  let response: Response;
  try {
    response = await fetchWithRetry(process.env.EMBEDDING_API_URL || "https://openrouter.ai/api/v1/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + apiKey },
      body: JSON.stringify(body),
    });
  } catch {
    throw new AppError("Could not reach the embedding provider. Please try again.", 502);
  }

  const responseText = await response.text();
  if (!response.ok) {
    console.error("Embedding request failed with status " + response.status);
    throw new AppError(
      "The embedding provider returned an error (" + response.status + "): " + readProviderMessage(responseText),
      502
    );
  }

  // Check the answer: one vector per text, each a list of finite numbers with the expected length.
  let items: { index?: number; embedding?: unknown }[] | undefined;
  try {
    items = JSON.parse(responseText).data;
  } catch {
    items = undefined;
  }
  if (!Array.isArray(items) || items.length !== texts.length) {
    throw new AppError("The embedding provider returned an unexpected response.", 502);
  }

  items.sort((a, b) => (a.index ?? 0) - (b.index ?? 0)); // providers may return them out of order
  const vectors: number[][] = [];
  for (const item of items) {
    const vector = item.embedding;
    const isNumberList =
      Array.isArray(vector) && vector.length > 0 && vector.every((n) => typeof n === "number" && Number.isFinite(n));
    if (!isNumberList) {
      throw new AppError("The embedding provider returned an invalid vector.", 502);
    }
    if (expectedLength > 0 && vector.length !== expectedLength) {
      throw new AppError(
        "The embedding has " + vector.length + " numbers but " + expectedLength + " were expected. Check EMBEDDING_MODEL and EMBEDDING_DIMENSIONS.",
        502
      );
    }
    vectors.push(vector);
  }
  return vectors;
}

// Turns many texts into many embeddings (same order). Used for the chunks of a document.
export async function generateEmbeddings(texts: string[]): Promise<number[][]> {
  const allVectors: number[][] = [];
  for (let start = 0; start < texts.length; start += EMBEDDING_BATCH_SIZE) {
    const batch = texts.slice(start, start + EMBEDDING_BATCH_SIZE);
    allVectors.push(...(await requestEmbeddings(batch)));
  }
  return allVectors;
}

// Turns one text into one embedding. Used for the user's question (same model as the chunks).
export async function generateEmbedding(text: string): Promise<number[]> {
  const vectors = await generateEmbeddings([text]);
  return vectors[0];
}

// ---------- Chat model ----------

// Sends a system prompt (the rules) and a user prompt to the chat model and returns its text reply.
export async function askChatModel(systemPrompt: string, userPrompt: string): Promise<string> {
  const apiKey = process.env.OPENROUTER_API_KEY || "";
  const model = process.env.OPENROUTER_MODEL || "";
  if (apiKey === "") {
    throw new AppError("OPENROUTER_API_KEY is not configured in backend/.env.", 503);
  }
  if (model === "") {
    throw new AppError("OPENROUTER_MODEL is not configured in backend/.env.", 503);
  }

  let response: Response;
  try {
    response = await fetchWithRetry(process.env.OPENROUTER_API_URL || "https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + apiKey },
      body: JSON.stringify({
        model: model,
        temperature: numberSetting("LLM_TEMPERATURE", 0), // low = less random wording (not a guarantee of facts)
        max_tokens: 800,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
      }),
    });
  } catch {
    throw new AppError("Could not reach the AI provider. Please try again.", 502);
  }

  const responseText = await response.text();
  if (!response.ok) {
    console.error("LLM request failed with status " + response.status);
    throw new AppError(
      "The AI provider returned an error (" + response.status + "): " + readProviderMessage(responseText),
      502
    );
  }

  let reply: unknown;
  try {
    reply = JSON.parse(responseText).choices?.[0]?.message?.content;
  } catch {
    reply = undefined;
  }
  if (typeof reply !== "string" || reply.trim() === "") {
    throw new AppError("The AI provider returned an empty answer.", 502);
  }
  return reply;
}
