// The building blocks of the page. Each one is a small React function that returns part of the screen.
// "props" are the values a parent (App) gives to a component. Functions passed as props (like onSend)
// are how a component tells its parent that something happened.
import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, KeyboardEvent } from "react";
import { uploadFile } from "./api";
import type { ChatSummary, Message, UploadedDocument } from "./api";

const MAX_FILE_SIZE_MB = 10;

// Checks a file in the browser. Returns an error message, or "" if the file is fine.
// (The backend checks again: the browser check is only for quick feedback.)
function checkFile(file: File): string {
  const name = file.name.toLowerCase();
  if (!name.endsWith(".pdf") && !name.endsWith(".txt")) {
    return "Unsupported file type. Please choose a PDF or TXT file.";
  }
  if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
    return "This file is larger than " + MAX_FILE_SIZE_MB + " MB. Please choose a smaller file.";
  }
  if (file.size === 0) {
    return "This file is empty. Please choose a different file.";
  }
  return "";
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// ======================================================================
// UPLOAD BOX: choose or drop a file, send it to the backend, show the real result
// ======================================================================
export function UploadBox({ onUploaded }: { onUploaded: (uploaded: UploadedDocument) => void }) {
  // useState keeps a value that React remembers. When we change it, the screen is redrawn.
  const [file, setFile] = useState<File | null>(null); // the chosen file
  const [result, setResult] = useState<UploadedDocument | null>(null); // the backend's answer
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null); // lets our button open the hidden file input

  // Used by both the "Select file" button and drag and drop.
  function chooseFile(newFile: File) {
    if (isUploading) return; // do not swap the file during an upload
    const problem = checkFile(newFile);
    if (problem !== "") {
      setError(problem);
      return; // an invalid file is not kept
    }
    setError("");
    setResult(null);
    setFile(newFile);
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0];
    if (chosen) chooseFile(chosen);
    event.target.value = ""; // so the same file can be chosen again later
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault(); // stop the browser from opening the file
    setIsDragging(false);
    if (event.dataTransfer.files.length > 1) {
      setError("Please drop one file at a time.");
      return;
    }
    const dropped = event.dataTransfer.files[0];
    if (dropped) chooseFile(dropped);
  }

  async function upload() {
    if (file === null || isUploading) return; // nothing to send, or already sending
    setIsUploading(true);
    setError("");
    try {
      const processed = await uploadFile(file);
      setResult(processed);
      setFile(null); // ready for the next file
      onUploaded(processed); // tell App, so the library and the selection are updated
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Something went wrong while uploading.");
    } finally {
      setIsUploading(false); // always stop the loading state, also after an error
    }
  }

  return (
    <section className="card">
      <h2>Add a document</h2>

      <div
        className={isDragging ? "drop-zone dragging" : "drop-zone"}
        onDrop={handleDrop}
        onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
      >
        <p className="drop-title">Drag and drop a PDF or TXT file here</p>
        <p className="drop-hint">or</p>
        <button type="button" className="button button-secondary" onClick={() => fileInput.current?.click()}>
          Select file
        </button>
        <p className="drop-hint">PDF or TXT, up to {MAX_FILE_SIZE_MB} MB</p>
        <input
          ref={fileInput}
          type="file"
          accept=".pdf,.txt,application/pdf,text/plain"
          className="hidden-input"
          aria-label="Choose a PDF or TXT file"
          onChange={handleInputChange}
        />
      </div>

      {file !== null && (
        <div className="file-row">
          <div className="file-info">
            <span className="file-name">{file.name}</span>
            <span className="file-size">{formatFileSize(file.size)}</span>
            <span className="status-label">
              {isUploading ? "Uploading, extracting text and creating embeddings…" : "Selected — ready for processing"}
            </span>
          </div>
          <div className="file-actions">
            <button type="button" className="button button-primary" onClick={upload} disabled={isUploading}>
              {isUploading ? "Processing…" : "Upload & process"}
            </button>
            <button type="button" className="button button-ghost" onClick={() => setFile(null)} disabled={isUploading}>
              Remove
            </button>
          </div>
        </div>
      )}

      {error !== "" && <p className="error-text" role="alert">{error}</p>}

      {result !== null && (
        <p className="success-text" role="status">
          {result.filename}: {result.status}. Extracted {result.characterCount.toLocaleString()} characters and
          created {result.chunkCount} chunk{result.chunkCount === 1 ? "" : "s"}.
          {result.warning !== undefined && " Note: " + result.warning}
        </p>
      )}
    </section>
  );
}

// ======================================================================
// DOCUMENT LIST: the user's library. Tick the documents a question should be about.
// ======================================================================
export function DocumentList(props: {
  documents: UploadedDocument[];
  selectedIds: string[];
  onToggle: (documentId: string) => void;
  onDelete: (document: UploadedDocument) => void;
}) {
  return (
    <section className="card">
      <h2>Your documents</h2>
      {props.documents.length === 0 && <p className="empty-text">No documents yet. Add one above.</p>}

      <ul className="item-list">
        {props.documents.map((document) => (
          <li className="item" key={document.id}>
            <label className="item-main">
              <input
                type="checkbox"
                checked={props.selectedIds.includes(document.id)}
                disabled={document.status !== "processed"}
                onChange={() => props.onToggle(document.id)}
                aria-label={"Ask about " + document.filename}
              />
              <span className="item-text">
                <span className="item-title">{document.filename}</span>
                <span className="item-meta">
                  {document.status === "processed" && document.chunkCount + (document.chunkCount === 1 ? " chunk" : " chunks") + " · " + document.fileType.toUpperCase()}
                  {document.status === "processing" && "Still processing…"}
                </span>
                {document.warning !== undefined && <span className="item-warning">{document.warning}</span>}
              </span>
            </label>
            <button
              type="button"
              className="button button-ghost"
              onClick={() => props.onDelete(document)}
              aria-label={"Delete " + document.filename}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

// ======================================================================
// CHAT LIST: the user's saved chats. Click one to reopen it.
// ======================================================================
export function ChatList(props: {
  chats: ChatSummary[];
  documents: UploadedDocument[];
  activeChatId: string | null;
  onOpen: (chat: ChatSummary) => void;
  onDelete: (chat: ChatSummary) => void;
  onNew: () => void;
}) {
  // Turns a list of document IDs into the file names, for example "a.pdf, b.txt".
  function documentNames(documentIds: string[]): string {
    return documentIds
      .map((id) => props.documents.find((document) => document.id === id)?.filename ?? "deleted document")
      .join(", ");
  }

  return (
    <section className="card">
      <div className="card-header">
        <h2>Your chats</h2>
        <button type="button" className="button button-secondary small-button" onClick={props.onNew}>
          New chat
        </button>
      </div>
      {props.chats.length === 0 && <p className="empty-text">No saved chats yet. Ask a question to start one.</p>}

      <ul className="item-list">
        {props.chats.map((chat) => (
          <li className={chat.id === props.activeChatId ? "item active" : "item"} key={chat.id}>
            <button type="button" className="item-main item-button" onClick={() => props.onOpen(chat)}>
              <span className="item-text">
                <span className="item-title">{chat.title}</span>
                <span className="item-meta">{documentNames(chat.documentIds)}</span>
              </span>
            </button>
            <button
              type="button"
              className="button button-ghost"
              onClick={() => props.onDelete(chat)}
              aria-label={"Delete chat " + chat.title}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

// ======================================================================
// MESSAGES
// ======================================================================

// One chat bubble. User messages go right, assistant messages go left.
function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  const sources = message.sources ?? [];

  return (
    <div className={isUser ? "message-row user-row" : "message-row assistant-row"}>
      <div className={isUser ? "bubble user-bubble" : "bubble assistant-bubble"}>
        <p className="bubble-text">{message.text}</p>
        {message.note && <p className="bubble-note">{message.note}</p>}

        {sources.length > 0 && (
          <div className="sources">
            <p className="sources-title">Sources</p>
            {sources.map((source, index) => (
              <div className="source" key={index}>
                <div className="source-header">
                  <span className="source-file">{source.filename}</span>
                  {source.pageNumber && <span className="source-page">Page {source.pageNumber}</span>}
                </div>
                <p className="source-excerpt">“{source.excerpt}”</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// The list of messages, with a "waiting" bubble while the answer is being made.
export function MessageList({ messages, isLoading }: { messages: Message[]; isLoading: boolean }) {
  const bottom = useRef<HTMLDivElement>(null); // an invisible element at the end of the list

  // useEffect runs after the screen changes. Here it scrolls down to the newest message.
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="chat-window" role="log" aria-label="Conversation" aria-live="polite">
      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}

      {isLoading && (
        <div className="message-row assistant-row">
          <div className="bubble assistant-bubble loading-bubble" role="status">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="loading-text">Waiting for a response…</span>
          </div>
        </div>
      )}

      <div ref={bottom}></div>
    </div>
  );
}

// ======================================================================
// QUESTION BOX: the textarea and the Send button
// ======================================================================
export function QuestionBox(props: { isLoading: boolean; isDisabled: boolean; onSend: (question: string) => void }) {
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");

  function send() {
    if (props.isDisabled || props.isLoading) return;
    const text = question.trim(); // remove spaces at the start and end
    if (text === "") {
      setError("Please type a question before sending.");
      return;
    }
    setError("");
    props.onSend(text);
    setQuestion(""); // clear the box
  }

  // Enter sends. Shift+Enter keeps the normal behaviour: a new line.
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // While typing with an input method (for example Japanese or Chinese), Enter confirms a word: it must not send.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    }
  }

  return (
    <div className="chat-input-area">
      {error !== "" && <p className="error-text" role="alert">{error}</p>}
      <div className="chat-input-row">
        <textarea
          className="chat-textarea"
          rows={2}
          maxLength={1000}
          aria-label="Your question"
          placeholder={props.isDisabled ? "Add a document and tick it to start asking…" : "Ask a question about the ticked documents…"}
          disabled={props.isDisabled}
          value={question}
          onChange={(event) => { setQuestion(event.target.value); setError(""); }}
          onKeyDown={handleKeyDown}
        />
        <button type="button" className="button button-primary" onClick={send} disabled={props.isLoading || props.isDisabled}>
          Send
        </button>
      </div>
      <p className="input-hint">Press Enter to send, Shift+Enter for a new line.</p>
    </div>
  );
}
