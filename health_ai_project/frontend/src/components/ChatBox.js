/**
 * MedAI – ChatBar Component
 * Fixed bottom chat input bar with file upload button.
 */

import React, { useRef, useState } from "react";

const ALLOWED_TYPES = [
  "application/pdf",
  "text/plain",
  "image/png",
  "image/jpeg",
  "image/jpg",
];

const styles = {
  bar: {
    position: "fixed",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "#fff",
    borderTop: "1px solid #e2e8f0",
    padding: "12px 16px",
    zIndex: 100,
    boxShadow: "0 -2px 10px rgba(0,0,0,0.06)",
  },
  inner: {
    maxWidth: 820,
    margin: "0 auto",
    display: "flex",
    alignItems: "flex-end",
    gap: 10,
  },
  inputWrapper: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  fileChip: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "4px 10px",
    backgroundColor: "#eef2ff",
    borderRadius: 6,
    fontSize: 12,
    color: "#4338ca",
    maxWidth: "fit-content",
  },
  fileRemove: {
    cursor: "pointer",
    fontWeight: 700,
    color: "#ef4444",
    border: "none",
    background: "none",
    fontSize: 14,
    padding: 0,
    lineHeight: 1,
  },
  textarea: {
    width: "100%",
    minHeight: 44,
    maxHeight: 120,
    padding: "10px 14px",
    fontSize: 15,
    borderRadius: 12,
    border: "2px solid #e2e8f0",
    fontFamily: "inherit",
    resize: "none",
    outline: "none",
    boxSizing: "border-box",
    lineHeight: 1.4,
    transition: "border-color 0.2s",
  },
  attachBtn: {
    width: 44,
    height: 44,
    borderRadius: 10,
    border: "2px solid #e2e8f0",
    backgroundColor: "#f8fafc",
    cursor: "pointer",
    fontSize: 20,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    transition: "background-color 0.2s",
  },
  sendBtn: {
    height: 44,
    padding: "0 20px",
    borderRadius: 10,
    border: "none",
    backgroundColor: "#4361ee",
    color: "#fff",
    fontSize: 15,
    fontWeight: 600,
    cursor: "pointer",
    flexShrink: 0,
    transition: "background-color 0.2s",
  },
  sendBtnDisabled: {
    backgroundColor: "#a0aec0",
    cursor: "not-allowed",
  },
};

export default function ChatBox({ onSubmit, loading, sidebarOpen, inline }) {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const fileRef = useRef(null);

  const canSend = !loading && (text.trim().length >= 1 || file);

  /* inline mode: rendered inside welcome screen, not fixed */
  const barStyle = inline
    ? {
        backgroundColor: "#fff",
        borderRadius: 16,
        padding: "10px 14px",
        boxShadow: "0 2px 12px rgba(0,0,0,0.08)",
        border: "1px solid #e2e8f0",
      }
    : {
        ...styles.bar,
        left: sidebarOpen ? 260 : 0,
        transition: "left 0.25s ease",
      };

  const handleSend = () => {
    if (!canSend) return;
    onSubmit(text.trim(), file);
    setText("");
    setFile(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileSelect = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    if (!ALLOWED_TYPES.includes(selected.type)) {
      alert("Unsupported file type. Please upload PDF, TXT, PNG, or JPG.");
      return;
    }
    if (selected.size > 10 * 1024 * 1024) {
      alert("File too large. Maximum size is 10 MB.");
      return;
    }
    setFile(selected);
    e.target.value = "";
  };

  return (
    <div style={barStyle}>
      <div style={styles.inner}>
        {/* Attach button */}
        <button
          style={styles.attachBtn}
          onClick={() => fileRef.current?.click()}
          title="Upload medical report"
          disabled={loading}
        >
          📎
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.png,.jpg,.jpeg"
          style={{ display: "none" }}
          onChange={handleFileSelect}
        />

        {/* Input area */}
        <div style={styles.inputWrapper}>
          {file && (
            <div style={styles.fileChip}>
              📄 {file.name}
              <button
                style={styles.fileRemove}
                onClick={() => setFile(null)}
                title="Remove file"
              >
                ✕
              </button>
            </div>
          )}
          <textarea
            style={{
              ...styles.textarea,
              borderColor: text ? "#4361ee" : "#e2e8f0",
            }}
            rows={1}
            placeholder="Describe your symptoms or ask a health question..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
        </div>

        {/* Send button */}
        <button
          style={{
            ...styles.sendBtn,
            ...(canSend ? {} : styles.sendBtnDisabled),
          }}
          onClick={handleSend}
          disabled={!canSend}
        >
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
