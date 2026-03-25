/**
 * MedAI – Full ChatGPT-Style Interface
 * Sidebar + multi-conversation management + user profile.
 * Wrapped with React Router and AuthContext for authentication.
 */

import React, { useState, useRef, useEffect, useCallback } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AuthProvider, { useAuth } from "./context/AuthContext";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import ProfilePage from "./pages/ProfilePage";
import Sidebar from "./components/Sidebar";
import ChatBox from "./components/ChatBox";
import ChatMessage from "./components/ChatMessage";
import { predictDisease } from "./services/api";

/* ── CSS keyframes (injected once) ───────────────────────── */
const injectStyles = (() => {
  let done = false;
  return () => {
    if (done) return;
    const sheet = document.createElement("style");
    sheet.textContent = `
      @keyframes bounce {
        0%,80%,100%{transform:translateY(0)}
        40%{transform:translateY(-6px)}
      }
      .medai-chat-area { scrollbar-width: none; -ms-overflow-style: none; }
      .medai-chat-area::-webkit-scrollbar { display: none; }
      .medai-main { transition: margin-left 0.25s ease; }
    `;
    document.head.appendChild(sheet);
    done = true;
  };
})();

/* ── localStorage helpers ────────────────────────────────── */
function getConversationStorageKey(user) {
  return user?.id ? `medai_conversations_user_${user.id}` : null;
}

function getActiveConversationStorageKey(user) {
  return user?.id ? `medai_active_conversation_user_${user.id}` : null;
}

function loadConversations(storageKey) {
  if (!storageKey) return [];
  try {
    const raw = localStorage.getItem(storageKey);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConversations(storageKey, convos) {
  if (!storageKey) return;
  try {
    localStorage.setItem(storageKey, JSON.stringify(convos));
  } catch { /* quota exceeded – silently ignore */ }
}

function loadActiveConversationId(storageKey) {
  if (!storageKey) return null;
  try {
    return localStorage.getItem(storageKey);
  } catch {
    return null;
  }
}

function saveActiveConversationId(storageKey, activeId) {
  if (!storageKey) return;
  try {
    if (activeId) {
      localStorage.setItem(storageKey, activeId);
    } else {
      localStorage.removeItem(storageKey);
    }
  } catch { /* ignore storage errors */ }
}

function restoreChatState(storageKey, activeStorageKey) {
  const saved = loadConversations(storageKey);
  const savedActiveId = loadActiveConversationId(activeStorageKey);

  if (saved.length > 0) {
    const restoredActiveId = saved.some((c) => c.id === savedActiveId)
      ? savedActiveId
      : saved[0].id;
    return {
      conversations: saved,
      activeId: restoredActiveId,
    };
  }

  const freshChat = createConversation();
  return {
    conversations: [freshChat],
    activeId: freshChat.id,
  };
}

/* ── generate a short title from first user message ──────── */
function generateTitle(text) {
  if (!text) return "New Chat";
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > 36 ? clean.slice(0, 36) + "…" : clean;
}

/* ── create a blank conversation object ──────────────────── */
function createConversation() {
  return {
    id: `chat_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    title: "New Chat",
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

/* ── Inline styles ───────────────────────────────────────── */
const s = {
  page: {
    display: "flex",
    height: "100vh",
    backgroundColor: "#f0f4f8",
    fontFamily: "'Segoe UI', 'Inter', Arial, sans-serif",
    overflow: "hidden",
  },
  main: (sidebarOpen) => ({
    flex: 1,
    display: "flex",
    flexDirection: "column",
    marginLeft: sidebarOpen ? 260 : 0,
    transition: "margin-left 0.25s ease",
    height: "100vh",
    overflow: "hidden",
  }),
  header: {
    background: "linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%)",
    padding: "14px 0 14px 52px",
    textAlign: "center",
    color: "#fff",
    boxShadow: "0 2px 8px rgba(67,97,238,0.25)",
    flexShrink: 0,
    zIndex: 10,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 800,
    margin: 0,
    letterSpacing: "-0.5px",
  },
  headerSub: {
    fontSize: 11,
    opacity: 0.85,
    margin: "2px 0 0",
    fontWeight: 400,
  },
  chatArea: (empty) => ({
    flex: 1,
    overflowY: "auto",
    padding: empty ? "0 20px" : "24px 20px 130px",
    maxWidth: 820,
    width: "100%",
    margin: "0 auto",
    boxSizing: "border-box",
    ...(empty
      ? {
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }
      : {}),
  }),
  welcome: {
    textAlign: "center",
    marginTop: 60,
    color: "#94a3b8",
  },
  welcomeIcon: { fontSize: 52, marginBottom: 12 },
  welcomeTitle: {
    fontSize: 22,
    fontWeight: 700,
    color: "#475569",
    margin: "0 0 8px",
  },
  welcomeText: {
    fontSize: 14,
    lineHeight: 1.7,
    maxWidth: 460,
    margin: "0 auto",
  },
  suggestionRow: {
    display: "flex",
    gap: 8,
    justifyContent: "center",
    marginTop: 24,
    flexWrap: "wrap",
  },
  suggestion: {
    padding: "9px 18px",
    backgroundColor: "#fff",
    border: "1px solid #cbd5e1",
    borderRadius: 20,
    fontSize: 13,
    color: "#475569",
    cursor: "pointer",
    transition: "all 0.15s",
  },
};

const SUGGESTIONS = [
  "I have a headache and fever",
  "Chest pain and shortness of breath",
  "Persistent cough for 2 weeks",
  "Skin rash and itching",
];

/* ── App ─────────────────────────────────────────────────── */

/* Protected route wrapper */
function RequireAuth({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return null;
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

/* Public route wrapper – redirect to / if already logged in */
function PublicOnly({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return null;
  return isAuthenticated ? <Navigate to="/" replace /> : children;
}

function ChatPage() {
  const { user } = useAuth();
  const storageKey = getConversationStorageKey(user);
  const activeStorageKey = getActiveConversationStorageKey(user);
  const initialChatState = restoreChatState(storageKey, activeStorageKey);

  const [conversations, setConversations] = useState(() => initialChatState.conversations);
  const [activeId, setActiveId] = useState(() => initialChatState.activeId);
  const [hydratedKey, setHydratedKey] = useState(() => storageKey);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const lastUserRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => { injectStyles(); }, []);

  useEffect(() => {
    const restored = restoreChatState(storageKey, activeStorageKey);
    setConversations(restored.conversations);
    setActiveId(restored.activeId);
    setHydratedKey(storageKey);
  }, [storageKey, activeStorageKey]);

  /* persist on every change – skip empty chats */
  useEffect(() => {
    if (!storageKey || hydratedKey !== storageKey) return;
    saveConversations(storageKey, conversations.filter((c) => c.messages.length > 0));
  }, [conversations, storageKey, hydratedKey]);

  useEffect(() => {
    if (!activeStorageKey || hydratedKey !== storageKey) return;
    saveActiveConversationId(activeStorageKey, activeId);
  }, [activeId, activeStorageKey, hydratedKey, storageKey]);

  /* scroll to latest user message */
  const activeConvo = conversations.find((c) => c.id === activeId);
  const messages = activeConvo?.messages || [];

  useEffect(() => {
    if (lastUserRef.current) {
      lastUserRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [messages.length, loading]);

  /* ── conversation management ───── */
  const handleNewChat = useCallback(() => {
    const c = createConversation();
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
  }, []);

  const handleSelectChat = useCallback((id) => {
    setActiveId(id);
  }, []);

  const handleDeleteChat = useCallback((id) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (id === activeId) {
        if (next.length > 0) {
          setActiveId(next[0].id);
        } else {
          const freshChat = createConversation();
          setActiveId(freshChat.id);
          return [freshChat];
        }
      }
      return next;
    });
  }, [activeId]);

  /* ── send handler ──────────────── */
  const handleSubmit = async (text, file) => {
    let targetId = activeId;

    /* If no active conversation, create one */
    if (!targetId) {
      const c = createConversation();
      setConversations((prev) => [c, ...prev]);
      setActiveId(c.id);
      targetId = c.id;
    }

    /* 1. Push user message */
    const userMsg = {
      id: Date.now(),
      role: "user",
      text: text || (file ? `📎 Uploaded report: ${file.name}` : ""),
      fileName: file?.name || null,
    };

    setConversations((prev) =>
      prev.map((c) =>
        c.id === targetId
          ? {
              ...c,
              messages: [...c.messages, userMsg],
              title: c.messages.length === 0 ? generateTitle(text || file?.name) : c.title,
              updatedAt: Date.now(),
            }
          : c
      )
    );
    setLoading(true);

    /* 2. Call API */
    try {
      const data = await predictDisease(text, file);

      /* If backend returned a greeting / conversational response */
      if (data.greeting) {
        const botMsg = {
          id: Date.now() + 1,
          role: "bot",
          type: "text",
          text: data.message,
        };
        setConversations((prev) =>
          prev.map((c) =>
            c.id === targetId
              ? { ...c, messages: [...c.messages, botMsg], updatedAt: Date.now() }
              : c
          )
        );
      } else if (data.summary_type === "report_summary") {
        /* Medical report summary response */
        const botMsg = {
          id: Date.now() + 1,
          role: "bot",
          type: "summary",
          data,
        };
        setConversations((prev) =>
          prev.map((c) =>
            c.id === targetId
              ? { ...c, messages: [...c.messages, botMsg], updatedAt: Date.now() }
              : c
          )
        );
      } else if (data.response_type === "report_followup" || data.response_type === "report_reset") {
        const botMsg = {
          id: Date.now() + 1,
          role: "bot",
          type: "text",
          text: data.message || data.response || "I couldn't find a report-based answer.",
        };
        setConversations((prev) =>
          prev.map((c) =>
            c.id === targetId
              ? { ...c, messages: [...c.messages, botMsg], updatedAt: Date.now() }
              : c
          )
        );
      } else {
        /* Normal prediction response */
        const botMsg = {
          id: Date.now() + 1,
          role: "bot",
          type: "prediction",
          data,
          uploadedFile: file?.name || null,
        };
        setConversations((prev) =>
          prev.map((c) =>
            c.id === targetId
              ? { ...c, messages: [...c.messages, botMsg], updatedAt: Date.now() }
              : c
          )
        );
      }
    } catch (err) {
      const errText =
        err.response?.data?.error ||
        err.response?.data?.details?.symptoms?.[0] ||
        err.message ||
        "Something went wrong. Please try again.";
      setConversations((prev) =>
        prev.map((c) =>
          c.id === targetId
            ? {
                ...c,
                messages: [
                  ...c.messages,
                  { id: Date.now() + 1, role: "bot", type: "error", text: errText },
                ],
                updatedAt: Date.now(),
              }
            : c
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestion = (txt) => handleSubmit(txt, null);

  const isEmpty = messages.length === 0 && !loading;

  return (
    <div style={s.page}>
      {/* ─── Sidebar ─── */}
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        collapsed={!sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
      />

      {/* ─── Main Panel ─── */}
      <div style={s.main(sidebarOpen)} className="medai-main">
        {/* Header */}
        <header style={s.header}>
          <h1 style={s.headerTitle}>🏥 MedAI</h1>
          <p style={s.headerSub}>AI-Powered Disease Prediction Chat</p>
        </header>

        {/* Chat thread */}
        <div className="medai-chat-area" style={s.chatArea(isEmpty)}>
          {/* Welcome screen – centered with ChatBox inline */}
          {isEmpty && (
            <div style={{ textAlign: "center", width: "100%", maxWidth: 680 }}>
              <div style={s.welcomeIcon}>🩺</div>
              <h2 style={s.welcomeTitle}>How can I help you today?</h2>
              <p style={s.welcomeText}>
                Describe your symptoms or upload a medical report.
              </p>

              {/* Inline ChatBox in center */}
              <div style={{ marginTop: 28 }}>
                <ChatBox
                  onSubmit={handleSubmit}
                  loading={loading}
                  sidebarOpen={sidebarOpen}
                  inline
                />
              </div>

              <div style={s.suggestionRow}>
                {SUGGESTIONS.map((txt) => (
                  <button
                    key={txt}
                    style={s.suggestion}
                    onClick={() => handleSuggestion(txt)}
                    onMouseEnter={(e) => {
                      e.target.style.backgroundColor = "#4361ee";
                      e.target.style.color = "#fff";
                      e.target.style.borderColor = "#4361ee";
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.backgroundColor = "#fff";
                      e.target.style.color = "#475569";
                      e.target.style.borderColor = "#cbd5e1";
                    }}
                  >
                    {txt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((msg, idx) => {
            const isLastUser =
              msg.role === "user" &&
              idx === messages.map((m) => m.role).lastIndexOf("user");
            return (
              <div key={msg.id} ref={isLastUser ? lastUserRef : undefined}>
                <ChatMessage message={msg} />
              </div>
            );
          })}

          {/* Typing indicator */}
          {loading && (
            <ChatMessage
              message={{ id: "loading", role: "bot", type: "loading" }}
            />
          )}

          <div ref={endRef} />
        </div>

        {/* Chat input bar – only show fixed bar when there are messages */}
        {!isEmpty && (
          <ChatBox
            onSubmit={handleSubmit}
            loading={loading}
            sidebarOpen={sidebarOpen}
          />
        )}
      </div>
    </div>
  );
}

/* ── Root App with Routing ───────────────────────────────── */
export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route
            path="/login"
            element={<PublicOnly><LoginPage /></PublicOnly>}
          />
          <Route
            path="/register"
            element={<PublicOnly><RegisterPage /></PublicOnly>}
          />
          <Route
            path="/profile"
            element={<RequireAuth><ProfilePage /></RequireAuth>}
          />
          <Route
            path="/"
            element={<RequireAuth><ChatPage /></RequireAuth>}
          />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
