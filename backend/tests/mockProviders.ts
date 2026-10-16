// TEST-ONLY local stand-in for the embedding and LLM providers.
//
// It lets us test OUR code (retrieval, validation, error handling) without an API key.
// It is NOT a real AI: "embeddings" are word counts and "answers" are copied sentences.
// It says nothing about how good a real model would be.
//
// Run alone for manual testing:  npx tsx scripts/mockProviders.ts   (port 5099)
import http from "http";

const STOP_WORDS = new Set([
  "the", "a", "an", "is", "are", "of", "in", "on", "to", "and", "what", "does", "do", "document", "say",
  "about", "how", "many", "which", "who", "was", "were", "for", "with", "by", "it", "this", "that", "at", "as",
  // words used by the test markers
  "mock", "fabricate", "chunk", "quote", "insufficient", "llm", "embed", "500", "unsupported", "claim", "fail", "once", "flaky", "json", "bad", "count", "wrong", "size", "null", "empty", "slow", "body", "choices",
]);

function tokenize(text: string): string[] {
  const words = text.toLowerCase().match(/[a-z0-9]+/g) || [];
  return words.filter((word) => !STOP_WORDS.has(word));
}

function hashWord(word: string): number {
  let hash = 0;
  for (let i = 0; i < word.length; i++) {
    hash = (hash * 31 + word.charCodeAt(i)) >>> 0;
  }
  return hash;
}

// "Embedding" = how many times each (hashed) word appears.
function fakeEmbedding(text: string, dimensions: number): number[] {
  const vector = new Array(dimensions).fill(0);
  for (const word of tokenize(text)) {
    vector[hashWord(word) % dimensions] += 1;
  }
  return vector;
}

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter((s) => s !== "");
}

export type MockStats = {
  embeddingCalls: number;
  answerCalls: number;
  verifierCalls: number;
  lastAnswerChunkTexts: string[]; // chunk texts the "LLM" received in its latest answer call
};

function answerFromContext(userPrompt: string, stats: MockStats): string {
  const question = (userPrompt.match(/^Question: (.*)$/m) || ["", ""])[1];

  const chunks: { id: string; text: string }[] = [];
  const chunkPattern = /<chunk id="(.*?)"[^>]*>\n([\s\S]*?)\n<\/chunk>/g;
  let found = chunkPattern.exec(userPrompt);
  while (found) {
    chunks.push({ id: found[1], text: found[2] });
    found = chunkPattern.exec(userPrompt);
  }
  stats.lastAnswerChunkTexts = chunks.map((c) => c.text);

  const isRetry = userPrompt.includes("Your previous reply was rejected");
  const fallback = JSON.stringify({ answer: "I couldn't find this information in the uploaded documents.", citations: [] });
  if (chunks.length === 0) return fallback;

  if (question.includes("[mock:insufficient]")) return fallback;
  if (question.includes("[mock:not-json]")) return "I think the answer is blue.";
  if (question.includes("[mock:fail-once]") && !isRetry) return "not json on the first try";
  if (question.includes("[mock:fabricate-chunk]")) {
    return JSON.stringify({ answer: "Made up answer.", citations: [{ chunkId: "made-up-id", quote: "something invented here" }] });
  }
  if (question.includes("[mock:fabricate-quote]")) {
    return JSON.stringify({
      answer: "Made up answer.",
      citations: [{ chunkId: chunks[0].id, quote: "this sentence is not in the chunk at all" }],
    });
  }

  // Default: pick the sentence that shares the most words with the question.
  const questionWords = new Set(tokenize(question));
  let bestSentence = "";
  let bestChunkId = "";
  let bestScore = 0;
  for (const chunk of chunks) {
    for (const sentence of splitSentences(chunk.text)) {
      const score = tokenize(sentence).filter((word) => questionWords.has(word)).length;
      if (score > bestScore) {
        bestScore = score;
        bestSentence = sentence;
        bestChunkId = chunk.id;
      }
    }
  }
  if (bestScore === 0) return fallback;

  const answerText = question.includes("[mock:unsupported-claim]") ? "UNSUPPORTED_CLAIM " + bestSentence : bestSentence;
  return JSON.stringify({ answer: answerText, citations: [{ chunkId: bestChunkId, quote: bestSentence }] });
}

function readBody(request: http.IncomingMessage): Promise<string> {
  return new Promise((resolve) => {
    let body = "";
    request.on("data", (piece) => (body += piece));
    request.on("end", () => resolve(body));
  });
}

export function startMockProviders(port: number): Promise<{ server: http.Server; baseUrl: string; stats: MockStats }> {
  let flakyEmbedCalls = 0; // how many requests containing "[mock:flaky-embed]" we have seen
  let flakyLlmCalls = 0; // same for "[mock:flaky-llm]"
  const stats: MockStats = { embeddingCalls: 0, answerCalls: 0, verifierCalls: 0, lastAnswerChunkTexts: [] };

  const server = http.createServer(async (request, response) => {
    const body = JSON.parse((await readBody(request)) || "{}");
    const sendJson = (status: number, data: unknown) => {
      response.writeHead(status, { "Content-Type": "application/json" });
      response.end(JSON.stringify(data));
    };

    if (request.url === "/embeddings") {
      stats.embeddingCalls++;
      const inputs: string[] = body.input;
      if (inputs.some((text) => text.includes("[mock:flaky-embed]"))) {
        flakyEmbedCalls++;
        if (flakyEmbedCalls <= 2) { // the first two requests fail, the third works (tests the retry)
          sendJson(500, { error: { message: "mock flaky embedding failure" } });
          return;
        }
      }
      // A misbehaving provider: answers with 200, but the content is wrong. Our code must notice.
      if (inputs.some((text) => text.includes("[mock:embed-not-json]"))) {
        response.writeHead(200, { "Content-Type": "text/plain" });
        response.end("this is not json");
        return;
      }
      if (inputs.some((text) => text.includes("[mock:embed-bad-count]"))) {
        sendJson(200, { object: "list", data: [] }); // fewer vectors than texts
        return;
      }
      if (inputs.some((text) => text.includes("[mock:embed-wrong-size]"))) {
        sendJson(200, { object: "list", data: inputs.map((_, index) => ({ index, embedding: [0.1, 0.2, 0.3] })) });
        return;
      }
      if (inputs.some((text) => text.includes("[mock:embed-null]"))) {
        sendJson(200, { object: "list", data: inputs.map((_, index) => ({ index, embedding: [null, 0.2, 0.3] })) });
        return;
      }
      if (inputs.some((text) => text.includes("[mock:embed-500]"))) {
        sendJson(500, { error: { message: "mock embedding failure" } });
        return;
      }
      const dimensions = body.dimensions || 64;
      const data = inputs.map((text, index) => ({ object: "embedding", index, embedding: fakeEmbedding(text, dimensions) }));
      sendJson(200, { object: "list", model: body.model, data });
      return;
    }

    if (request.url === "/chat") {
      const systemPrompt: string = body.messages[0].content;
      const userPrompt: string = body.messages[1].content;
      const reply = (content: string) => sendJson(200, { choices: [{ message: { role: "assistant", content } }] });

      if (systemPrompt.includes("claim verifier")) {
        stats.verifierCalls++;
        reply(JSON.stringify({ supported: !userPrompt.includes("UNSUPPORTED_CLAIM") }));
        return;
      }

      stats.answerCalls++;
      if (userPrompt.includes("[mock:llm-empty]")) {
        reply(""); // an empty answer
        return;
      }
      if (userPrompt.includes("[mock:llm-not-json-body]")) {
        response.writeHead(200, { "Content-Type": "text/plain" });
        response.end("this is not json");
        return;
      }
      if (userPrompt.includes("[mock:llm-no-choices]")) {
        sendJson(200, {}); // no "choices" at all
        return;
      }
      if (userPrompt.includes("[mock:slow]")) {
        await new Promise((resolve) => setTimeout(resolve, 700)); // an answer that takes a while (tests deleting mid-answer)
      }
      if (userPrompt.includes("[mock:flaky-llm]")) {
        flakyLlmCalls++;
        if (flakyLlmCalls <= 2) { // the first two requests fail, the third works (tests the retry)
          sendJson(500, { error: { message: "mock flaky llm failure" } });
          return;
        }
      }
      if (userPrompt.includes("[mock:llm-500]")) {
        sendJson(500, { error: { message: "mock llm failure" } });
        return;
      }
      reply(answerFromContext(userPrompt, stats));
      return;
    }

    sendJson(404, { error: { message: "unknown mock route" } });
  });

  return new Promise((resolve) => {
    server.listen(port, () => {
      const address = server.address() as { port: number };
      resolve({ server, baseUrl: "http://localhost:" + address.port, stats });
    });
  });
}

// When run directly: start on port 5099 and keep running.
if (require.main === module) {
  startMockProviders(5099).then(({ baseUrl }) => {
    console.log("Mock providers running. Embeddings: " + baseUrl + "/embeddings   Chat: " + baseUrl + "/chat");
  });
}
