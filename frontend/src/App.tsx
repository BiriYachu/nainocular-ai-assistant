// The app: it keeps the shared state (documents, chats, messages) and puts the components together.
import { useEffect, useRef, useState } from "react";
import { askBackend, deleteChat, deleteDocument, listChats, listDocuments, loadChatMessages } from "./api";
import type { ChatSummary, Message, UploadedDocument } from "./api";
import { ChatList, DocumentList, MessageList, QuestionBox, UploadBox } from "./components";

const MAX_DOCUMENTS_PER_QUESTION = 5; // the backend has the same limit

// Turns whatever went wrong into a text we can show.
function readMessage(problem: unknown): string {
  return problem instanceof Error ? problem.message : "Something went wrong. Please try again.";
}

function App() {
  // State: values React remembers. Changing them with their "set" function redraws the screen.
  const [documents, setDocuments] = useState<UploadedDocument[]>([]); // the user's library
  const [chats, setChats] = useState<ChatSummary[]>([]); // the user's saved chats
  const [selectedIds, setSelectedIds] = useState<string[]>([]); // the ticked documents: what questions are about
  const [messages, setMessages] = useState<Message[]>([]); // the open chat (empty until the first question)
  const [conversationId, setConversationId] = useState<string | null>(null); // the saved chat that is open
  const [isLoading, setIsLoading] = useState(false); // waiting for an answer
  const [isOpeningChat, setIsOpeningChat] = useState(false); // loading an old chat
  const [error, setError] = useState("");
  // Counts how often the open chat changed. A slow request that finishes AFTER the user moved on
  // to another chat sees that the number changed and does not touch the screen.
  const chatVersion = useRef(0);

  // Reads the library and the chat list from the backend.
  async function refreshLists() {
    try {
      const [newDocuments, newChats] = await Promise.all([listDocuments(), listChats()]);
      setDocuments(newDocuments);
      setChats(newChats);
      return { newDocuments, newChats };
    } catch (problem) {
      setError(readMessage(problem));
      return null;
    }
  }

  // When the page opens, load what this browser already has (so nothing needs to be uploaded again).
  useEffect(() => {
    refreshLists();
  }, []);

  // Starts an empty chat about the given documents.
  function startNewChat(newSelection: string[]) {
    chatVersion.current += 1;
    setSelectedIds(newSelection);
    setConversationId(null);
    setMessages([]);
    setIsOpeningChat(false);
    setError("");
  }

  // UploadBox calls this when a file was processed.
  function handleUploaded(uploaded: UploadedDocument) {
    startNewChat(uploaded.status === "processed" ? [uploaded.id] : []);
    refreshLists();
  }

  // Ticking or unticking a document starts a new chat, because a chat belongs to its documents.
  function handleToggleDocument(documentId: string) {
    if (selectedIds.includes(documentId)) {
      startNewChat(selectedIds.filter((id) => id !== documentId));
      return;
    }
    if (selectedIds.length >= MAX_DOCUMENTS_PER_QUESTION) {
      setError("You can ask about at most " + MAX_DOCUMENTS_PER_QUESTION + " documents at once.");
      return;
    }
    startNewChat([...selectedIds, documentId]);
  }

  async function handleDeleteDocument(document: UploadedDocument) {
    if (!window.confirm("Delete " + document.filename + " and every chat that used it?")) return;
    try {
      await deleteDocument(document.id);
    } catch (problem) {
      setError(readMessage(problem));
      return;
    }
    const lists = await refreshLists();
    // The backend also deleted the chats that used this document. Close the open chat if it was one of them.
    const openChatIsGone = lists !== null && conversationId !== null && !lists.newChats.some((chat) => chat.id === conversationId);
    if (selectedIds.includes(document.id) || openChatIsGone) {
      startNewChat(selectedIds.filter((id) => id !== document.id));
    }
  }

  // Opens a saved chat: shows its messages and ticks its documents.
  async function handleOpenChat(chat: ChatSummary) {
    chatVersion.current += 1;
    const thisVersion = chatVersion.current;
    setError("");
    setIsOpeningChat(true);
    try {
      const savedMessages = await loadChatMessages(chat.id);
      if (thisVersion !== chatVersion.current) return; // the user already moved on
      setSelectedIds(chat.documentIds.filter((id) => documents.some((document) => document.id === id)));
      setConversationId(chat.id);
      setMessages(savedMessages);
    } catch (problem) {
      if (thisVersion === chatVersion.current) setError(readMessage(problem));
    } finally {
      if (thisVersion === chatVersion.current) setIsOpeningChat(false);
    }
  }

  async function handleDeleteChat(chat: ChatSummary) {
    if (!window.confirm('Delete the chat "' + chat.title + '"?')) return;
    try {
      await deleteChat(chat.id);
    } catch (problem) {
      setError(readMessage(problem));
      return;
    }
    if (chat.id === conversationId) startNewChat(selectedIds);
    refreshLists();
  }

  // Called by QuestionBox when the user sends a question.
  async function handleSend(question: string) {
    if (selectedIds.length === 0) {
      setError("Please tick at least one document first.");
      return;
    }
    setError("");
    const thisVersion = chatVersion.current;
    setMessages((old) => [...old, { role: "user", text: question }]); // show the question at once
    setIsLoading(true);

    try {
      const result = await askBackend(question, selectedIds, conversationId);
      refreshLists(); // the chat (new or updated) shows up in the list
      if (thisVersion !== chatVersion.current) return; // the user opened another chat meanwhile: it is saved, not shown here
      setConversationId(result.conversationId); // the first answer creates the saved chat
      setMessages((old) => [...old, { role: "assistant", text: result.answer, note: result.note, sources: result.sources }]);
    } catch (problem) {
      if (thisVersion === chatVersion.current) setError(readMessage(problem));
    } finally {
      setIsLoading(false); // always hide the loading bubble
    }
  }

  const selectedNames = documents.filter((document) => selectedIds.includes(document.id)).map((document) => document.filename);

  return (
    <div className="app">
      <header className="header">
        <div className="logo-dot" aria-hidden="true"></div>
        <div>
          <h1>NainOcular AI Assistant</h1>
          <p>Ask questions about your documents.</p>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <UploadBox onUploaded={handleUploaded} />
          <DocumentList
            documents={documents}
            selectedIds={selectedIds}
            onToggle={handleToggleDocument}
            onDelete={handleDeleteDocument}
          />
          <ChatList
            chats={chats}
            documents={documents}
            activeChatId={conversationId}
            onOpen={handleOpenChat}
            onDelete={handleDeleteChat}
            onNew={() => startNewChat(selectedIds)}
          />
          <p className="privacy-note">
            Your documents and chats are private to this browser. Opening the app in another browser, or clearing this
            browser's data, shows an empty library.
          </p>
        </aside>

        <section className="card chat-card">
          <p className="chat-scope">
            {selectedNames.length === 0 ? "No document ticked yet." : "Asking about: " + selectedNames.join(", ")}
          </p>
          <MessageList messages={messages} isLoading={isLoading} />
          {error !== "" && <p className="error-text chat-error" role="alert">{error}</p>}
          <QuestionBox isLoading={isLoading} isDisabled={selectedIds.length === 0 || isOpeningChat} onSend={handleSend} />
        </section>
      </div>
    </div>
  );
}

export default App;
