/**
 * MedAI – Sidebar Component
 * ChatGPT-style sidebar with new chat, chat history, and user profile.
 */

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

/* ── styles ──────────────────────────────────────────────── */
const s = {
  overlay: {
    position: "fixed",
    inset: 0,
    backgroundColor: "rgba(0,0,0,0.3)",
    zIndex: 199,
  },
  sidebar: (collapsed) => ({
    position: "fixed",
    top: 0,
    left: 0,
    bottom: 0,
    width: collapsed ? 0 : 260,
    backgroundColor: "#1e1e2e",
    color: "#e2e8f0",
    display: "flex",
    flexDirection: "column",
    zIndex: 200,
    transition: "width 0.25s ease",
    overflow: "hidden",
    boxShadow: collapsed ? "none" : "2px 0 12px rgba(0,0,0,0.25)",
  }),
  /* top section */
  topSection: {
    padding: "16px 14px 8px",
    flexShrink: 0,
  },
  newChatBtn: {
    width: "100%",
    padding: "10px 14px",
    border: "1px solid #444",
    borderRadius: 8,
    backgroundColor: "transparent",
    color: "#e2e8f0",
    fontSize: 14,
    fontWeight: 600,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: 8,
    transition: "background-color 0.15s",
  },
  /* search */
  searchBox: {
    margin: "10px 14px 6px",
    padding: "8px 10px",
    borderRadius: 6,
    border: "1px solid #333",
    backgroundColor: "#2a2a3d",
    color: "#e2e8f0",
    fontSize: 13,
    outline: "none",
    width: "calc(100% - 28px)",
    boxSizing: "border-box",
  },
  /* history list */
  historySection: {
    flex: 1,
    overflowY: "auto",
    padding: "4px 0",
    scrollbarWidth: "none",
    msOverflowStyle: "none",
  },
  groupLabel: {
    fontSize: 11,
    fontWeight: 700,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.6px",
    padding: "12px 16px 4px",
  },
  chatItem: (active) => ({
    padding: "9px 16px",
    fontSize: 13,
    color: active ? "#fff" : "#b0b8c8",
    backgroundColor: active ? "#333355" : "transparent",
    cursor: "pointer",
    borderRadius: 6,
    margin: "1px 8px",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
    display: "flex",
    alignItems: "center",
    gap: 8,
    transition: "background-color 0.12s",
  }),
  deleteBtn: {
    marginLeft: "auto",
    background: "none",
    border: "none",
    color: "#ef4444",
    cursor: "pointer",
    fontSize: 14,
    padding: "0 2px",
    opacity: 0.6,
    flexShrink: 0,
  },
  /* bottom user section */
  userSection: {
    borderTop: "1px solid #333",
    padding: "10px 14px",
    flexShrink: 0,
  },
  userTop: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    cursor: "pointer",
    padding: "4px 0",
    borderRadius: 6,
    transition: "background-color 0.12s",
  },
  userAvatar: {
    width: 34,
    height: 34,
    borderRadius: "50%",
    background: "linear-gradient(135deg, #4361ee, #7c3aed)",
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 14,
    fontWeight: 700,
    flexShrink: 0,
  },
  userName: {
    fontSize: 14,
    fontWeight: 600,
    color: "#e2e8f0",
    margin: 0,
    lineHeight: 1.2,
  },
  userPlan: {
    fontSize: 11,
    color: "#64748b",
    margin: 0,
  },
  userActions: {
    display: "flex",
    gap: 6,
    marginTop: 8,
  },
  userBtn: {
    flex: 1,
    padding: "7px 0",
    border: "1px solid #333",
    borderRadius: 6,
    backgroundColor: "transparent",
    color: "#b0b8c8",
    fontSize: 12,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.12s",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 4,
  },
  /* toggle button (open sidebar) - shown when collapsed */
  openBtn: {
    position: "fixed",
    top: 16,
    left: 16,
    zIndex: 201,
    width: 34,
    height: 34,
    borderRadius: 8,
    border: "none",
    backgroundColor: "transparent",
    color: "#475569",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "background-color 0.15s",
  },
  /* close sidebar button - shown inside sidebar top */
  closeBtn: {
    width: 34,
    height: 34,
    borderRadius: 8,
    border: "none",
    backgroundColor: "transparent",
    color: "#94a3b8",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    transition: "background-color 0.15s",
  },
  /* new chat icon button inside sidebar top row */
  newChatIcon: {
    width: 34,
    height: 34,
    borderRadius: 8,
    border: "none",
    backgroundColor: "transparent",
    color: "#94a3b8",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginLeft: "auto",
    flexShrink: 0,
    transition: "background-color 0.15s",
  },
  topRow: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "12px 10px 4px",
  },
};

/* ── group chats by time ─────────────────────────────────── */
function groupChats(conversations) {
  const now = Date.now();
  const DAY = 86400000;
  const groups = { Today: [], Yesterday: [], "Previous 7 Days": [], Older: [] };

  conversations.forEach((c) => {
    const age = now - c.updatedAt;
    if (age < DAY) groups["Today"].push(c);
    else if (age < 2 * DAY) groups["Yesterday"].push(c);
    else if (age < 7 * DAY) groups["Previous 7 Days"].push(c);
    else groups["Older"].push(c);
  });

  return Object.entries(groups).filter(([, items]) => items.length > 0);
}

export default function Sidebar({
  conversations,
  activeId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  collapsed,
  onToggle,
}) {
  const [search, setSearch] = useState("");
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const filtered = search.trim()
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(search.toLowerCase())
      )
    : conversations;
  const grouped = groupChats(filtered);

  const displayName = user
    ? [user.first_name, user.last_name].filter(Boolean).join(" ") || user.username
    : "User";
  const avatarInitial = user?.avatar_initial || displayName[0]?.toUpperCase() || "U";

  const handleLogout = async () => {
    await logout();
    navigate("/login", { replace: true });
  };

  /* SVG icons matching ChatGPT */
  const SidebarIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <line x1="9" y1="3" x2="9" y2="21" />
    </svg>
  );
  const PenIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z" />
    </svg>
  );

  return (
    <>
      {/* Open sidebar button – only visible when collapsed */}
      {collapsed && (
        <button
          style={s.openBtn}
          onClick={onToggle}
          title="Open sidebar"
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#e2e8f0")}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
        >
          <SidebarIcon />
        </button>
      )}

      {/* Mobile overlay */}
      {!collapsed && window.innerWidth < 768 && (
        <div style={s.overlay} onClick={onToggle} />
      )}

      {/* Sidebar panel */}
      <div style={s.sidebar(collapsed)}>
        {/* Top row: close sidebar + new chat */}
        <div style={s.topRow}>
          <button
            style={s.closeBtn}
            onClick={onToggle}
            title="Close sidebar"
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#2a2a3d")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            <SidebarIcon />
          </button>
          <button
            style={s.newChatIcon}
            onClick={onNewChat}
            title="New chat"
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#2a2a3d")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            <PenIcon />
          </button>
        </div>

        {/* New Chat text button */}
        <div style={s.topSection}>
          <button
            style={s.newChatBtn}
            onClick={onNewChat}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#2a2a3d")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            ✏️ New Chat
          </button>
        </div>

        {/* Search */}
        <input
          style={s.searchBox}
          placeholder="🔍 Search chats..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />

        {/* Chat History */}
        <div style={s.historySection} className="medai-chat-area">
          {grouped.length === 0 && (
            <div style={{ padding: "20px 16px", color: "#64748b", fontSize: 13, textAlign: "center" }}>
              No conversations yet
            </div>
          )}
          {grouped.map(([label, items]) => (
            <div key={label}>
              <div style={s.groupLabel}>{label}</div>
              {items.map((c) => (
                <div
                  key={c.id}
                  style={s.chatItem(c.id === activeId)}
                  onClick={() => onSelectChat(c.id)}
                  onMouseEnter={(e) => {
                    if (c.id !== activeId) e.currentTarget.style.backgroundColor = "#2a2a3d";
                  }}
                  onMouseLeave={(e) => {
                    if (c.id !== activeId) e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  💬 <span style={{ overflow: "hidden", textOverflow: "ellipsis", flex: 1 }}>{c.title}</span>
                  <button
                    style={s.deleteBtn}
                    onClick={(e) => { e.stopPropagation(); onDeleteChat(c.id); }}
                    title="Delete chat"
                    onMouseEnter={(e) => (e.currentTarget.style.opacity = 1)}
                    onMouseLeave={(e) => (e.currentTarget.style.opacity = 0.6)}
                  >
                    🗑
                  </button>
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* User Profile */}
        <div style={s.userSection}>
          <div
            style={s.userTop}
            onClick={() => navigate("/profile")}
            title="View profile"
          >
            <div style={s.userAvatar}>{avatarInitial}</div>
            <div>
              <p style={s.userName}>{displayName}</p>
              <p style={s.userPlan}>MedAI User</p>
            </div>
          </div>
          <div style={s.userActions}>
            <button
              style={s.userBtn}
              onClick={() => navigate("/profile")}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#2a2a3d";
                e.currentTarget.style.color = "#e2e8f0";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "transparent";
                e.currentTarget.style.color = "#b0b8c8";
              }}
            >
              👤 Profile
            </button>
            <button
              style={{ ...s.userBtn, borderColor: "#442222" }}
              onClick={handleLogout}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#3d2020";
                e.currentTarget.style.color = "#ef4444";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "transparent";
                e.currentTarget.style.color = "#b0b8c8";
                e.currentTarget.style.borderColor = "#442222";
              }}
            >
              🚪 Logout
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
