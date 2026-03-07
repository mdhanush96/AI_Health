/**
 * MedAI – ChatMessage Component
 * Human-centric, empathetic conversational rendering.
 */

import React from "react";

/* ── risk emoji + label helpers ──────────────────────────── */
const riskMeta = {
  "High Probability": { emoji: "🔴", color: "#ef4444", bg: "#fef2f2", text: "#991b1b" },
  "Moderate Probability": { emoji: "🟡", color: "#f59e0b", bg: "#fffbeb", text: "#92400e" },
  "Low Confidence": { emoji: "🟢", color: "#3b82f6", bg: "#eff6ff", text: "#1e40af" },
};

/* ── styles ──────────────────────────────────────────────── */
const s = {
  /* row layout */
  row: (isUser) => ({
    display: "flex",
    justifyContent: isUser ? "flex-end" : "flex-start",
    marginBottom: 24,
    gap: 12,
    alignItems: "flex-start",
  }),
  avatar: {
    width: 32,
    height: 32,
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 15,
    flexShrink: 0,
    fontWeight: 700,
  },
  botAvatar: {
    background: "linear-gradient(135deg, #4361ee, #3a0ca3)",
    color: "#fff",
  },
  userAvatar: {
    backgroundColor: "#e2e8f0",
    color: "#475569",
  },

  /* user bubble */
  userBubble: {
    maxWidth: "70%",
    padding: "10px 16px",
    borderRadius: "18px 18px 4px 18px",
    backgroundColor: "#4361ee",
    color: "#fff",
    fontSize: 15,
    lineHeight: 1.55,
    wordBreak: "break-word",
  },
  fileTag: {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    padding: "3px 8px",
    backgroundColor: "rgba(255,255,255,0.18)",
    borderRadius: 6,
    fontSize: 12,
    marginTop: 6,
  },

  /* bot body – no bubble, just flowing text like ChatGPT */
  botBody: {
    flex: 1,
    maxWidth: "85%",
    fontSize: 15,
    lineHeight: 1.7,
    color: "#1a1a2e",
    wordBreak: "break-word",
  },

  /* typing dots */
  typing: {
    display: "inline-flex",
    gap: 5,
    padding: "4px 0",
  },
  dot: (delay) => ({
    width: 8,
    height: 8,
    borderRadius: "50%",
    backgroundColor: "#94a3b8",
    animation: "bounce 1.2s infinite ease-in-out",
    animationDelay: `${delay}s`,
  }),

  /* conversational text blocks */
  intro: {
    margin: "0 0 14px",
    fontSize: 15,
    lineHeight: 1.7,
    color: "#374151",
  },
  sectionSep: {
    height: 1,
    backgroundColor: "#e5e7eb",
    border: "none",
    margin: "16px 0",
  },
  predBlock: {
    margin: "0 0 18px",
  },
  predTitle: {
    fontSize: 15,
    fontWeight: 700,
    margin: "0 0 4px",
    color: "#1a1a2e",
    textTransform: "capitalize",
  },

  infoText: {
    fontSize: 14,
    color: "#4b5563",
    lineHeight: 1.6,
    margin: "4px 0",
  },
  symptomsLine: {
    fontSize: 13,
    color: "#6b7280",
    margin: "4px 0",
  },
  adviceLine: {
    fontSize: 14,
    color: "#059669",
    margin: "6px 0 0",
  },
  emergencyBlock: {
    padding: "14px 18px",
    backgroundColor: "#fffbeb",
    border: "1px solid #fde68a",
    borderRadius: 10,
    marginBottom: 16,
    fontSize: 14,
    color: "#78350f",
    fontWeight: 400,
    whiteSpace: "pre-line",
    lineHeight: 1.75,
  },
  emergencyExplanation: {
    margin: "0 0 10px",
    fontSize: 14,
    lineHeight: 1.7,
    color: "#92400e",
  },
  emergencyHelplines: {
    margin: "10px 0 8px",
    padding: "10px 14px",
    backgroundColor: "#fef2f2",
    borderRadius: 8,
    border: "1px solid #fca5a5",
    fontSize: 13,
    lineHeight: 1.8,
    color: "#991b1b",
    fontWeight: 500,
    whiteSpace: "pre-line",
  },
  emergencyAction: {
    margin: "8px 0 0",
    fontSize: 14,
    fontWeight: 600,
    color: "#dc2626",
  },
  disclaimerBlock: {
    marginTop: 14,
    padding: "8px 12px",
    backgroundColor: "#f9fafb",
    borderLeft: "3px solid #d1d5db",
    borderRadius: 4,
    fontSize: 12,
    color: "#6b7280",
    fontStyle: "italic",
  },
  kbSection: {
    margin: "8px 0",
    padding: "10px 14px",
    backgroundColor: "#f8fafc",
    borderRadius: 8,
    borderLeft: "3px solid #4361ee",
  },
  kbSectionTitle: {
    fontSize: 14,
    fontWeight: 700,
    color: "#1e293b",
    margin: "0 0 6px",
  },
  kbSectionText: {
    fontSize: 13,
    color: "#475569",
    lineHeight: 1.7,
    margin: 0,
    whiteSpace: "pre-line",
  },
  confidenceBadge: (rm) => ({
    display: "inline-block",
    padding: "2px 10px",
    borderRadius: 12,
    fontSize: 12,
    fontWeight: 600,
    backgroundColor: rm.bg,
    color: rm.text,
    marginLeft: 8,
  }),
};

/* ── human intro sentences ───────────────────────────────── */
const EMPATHETIC_INTROS = [
  "I understand you're not feeling well. Let me help you make sense of your symptoms.",
  "Thank you for sharing your symptoms with me. Here's what I found that might help.",
  "I've carefully looked at your symptoms. Let me walk you through what this could be.",
  "I can see you're concerned about your health. Let me share what my analysis suggests.",
];

function getEmpathyIntro(uploadedFile) {
  if (uploadedFile) {
    return "I've carefully reviewed your uploaded report. Here's what I found:";
  }
  return EMPATHETIC_INTROS[Math.floor(Math.random() * EMPATHETIC_INTROS.length)];
}

/* ── KB section renderer ─────────────────────────────────── */
function KBSection({ icon, title, content }) {
  if (!content) return null;
  return (
    <div style={s.kbSection}>
      <p style={s.kbSectionTitle}>{icon} {title}</p>
      <p style={s.kbSectionText}>{content}</p>
    </div>
  );
}

/* ── conversational prediction renderer ──────────────────── */
function ConversationalResponse({ data, uploadedFile }) {
  const { predictions = [], emergency, disclaimer } = data;
  const top = predictions[0];
  const rest = predictions.slice(1);
  const kb = top?.info?.knowledge_base;

  return (
    <div>
      {/* Emergency — caring explanation first, then helplines */}
      {emergency?.is_emergency && (
        <div style={s.emergencyBlock}>
          <p style={s.emergencyExplanation}>
            ⚠️ {emergency.explanation || "The symptoms you described may need urgent medical attention."}
          </p>
          <div style={s.emergencyHelplines}>
            {emergency.helplines || (
              "📞 Emergency Helplines:\n• 112 — Emergency Number\n• 108 — Ambulance"
            )}
          </div>
          <p style={s.emergencyAction}>
            🏥 {emergency.message}
          </p>
        </div>
      )}

      {/* Empathetic intro */}
      <p style={s.intro}>{getEmpathyIntro(uploadedFile)}</p>

      {/* Top prediction – full detail */}
      {top && (
        <div style={s.predBlock}>
          <p style={{ ...s.predTitle, fontSize: 16 }}>
            Based on your symptoms, this looks like it could be{" "}
            <span style={{ color: "#4361ee" }}>{top.disease.replace(/_/g, " ")}</span>
            {(() => {
              const rm = riskMeta[top.risk_level] || riskMeta["Low Confidence"];
              return <span style={s.confidenceBadge(rm)}>{rm.emoji} {top.confidence}%</span>;
            })()}
          </p>

          {/* KB-powered detailed sections */}
          {kb ? (
            <>
              <KBSection icon="📋" title="What is it?" content={kb.overview} />
              <KBSection icon="🩹" title="Common Symptoms" content={kb.symptoms} />
              <KBSection icon="🔬" title="Possible Causes" content={kb.causes} />
              <KBSection icon="💊" title="Treatment & Care" content={kb.treatment} />
              <KBSection icon="⚠️" title="When to See a Doctor" content={kb.when_to_see_a_doctor} />
              {kb.preventions && (
                <KBSection icon="🛡️" title="Prevention Tips" content={kb.preventions} />
              )}
            </>
          ) : (
            <>
              {top.info?.description && (
                <p style={s.infoText}>{top.info.description}</p>
              )}
              {top.info?.common_symptoms?.length > 0 && (
                <p style={s.symptomsLine}>
                  🩹 Common symptoms: {top.info.common_symptoms.join(", ")}
                </p>
              )}
              {top.info?.advice && (
                <p style={s.adviceLine}>💡 <em>{top.info.advice}</em></p>
              )}
            </>
          )}
        </div>
      )}

      {/* Other possibilities */}
      {rest.length > 0 && (
        <>
          <hr style={s.sectionSep} />
          <p style={{ ...s.intro, marginBottom: 10, fontWeight: 600 }}>
            I also considered these other possibilities:
          </p>
          {rest.map((pred, idx) => {
            const rm = riskMeta[pred.risk_level] || riskMeta["Low Confidence"];
            const pkb = pred.info?.knowledge_base;
            return (
              <div key={`${pred.disease}-${idx}`} style={{ ...s.predBlock, marginBottom: 14 }}>
                <p style={s.predTitle}>
                  {idx + 2}. {pred.disease.replace(/_/g, " ")}
                  <span style={s.confidenceBadge(rm)}>{rm.emoji} {pred.confidence}%</span>
                </p>

                {pkb?.overview ? (
                  <p style={s.infoText}>{pkb.overview}</p>
                ) : pred.info?.description ? (
                  <p style={s.infoText}>{pred.info.description}</p>
                ) : null}

                {!pkb && pred.info?.common_symptoms?.length > 0 && (
                  <p style={s.symptomsLine}>
                    🩹 Common symptoms: {pred.info.common_symptoms.join(", ")}
                  </p>
                )}
              </div>
            );
          })}
        </>
      )}

      {/* Warm closing */}
      <hr style={s.sectionSep} />
      <p style={s.intro}>
        I hope this helps you understand what you might be experiencing. 💙
        Please remember, I'm here to guide — not to diagnose. If your symptoms
        persist or get worse, <strong>please visit a doctor</strong> for a proper evaluation.
        Your health matters, and getting professional advice is always the best step. 🙏
      </p>

      {disclaimer && (
        <div style={s.disclaimerBlock}>⚕️ {disclaimer}</div>
      )}
    </div>
  );
}

/* ── simple markdown renderer ───────────────────────────── */
function renderMarkdown(text) {
  if (!text) return null;
  const lines = text.split("\n");
  return lines.map((line, i) => {
    // Horizontal rule
    if (/^---+$/.test(line.trim())) {
      return <hr key={i} style={s.sectionSep} />;
    }
    // Process inline markdown
    const parts = [];
    let remaining = line;
    let idx = 0;
    // Bold: **text**
    const boldRegex = /\*\*(.+?)\*\*/g;
    let lastEnd = 0;
    let match;
    while ((match = boldRegex.exec(remaining)) !== null) {
      if (match.index > lastEnd) {
        parts.push(remaining.slice(lastEnd, match.index));
      }
      parts.push(<strong key={`b${i}-${idx++}`}>{match[1]}</strong>);
      lastEnd = match.index + match[0].length;
    }
    if (lastEnd < remaining.length) {
      let tail = remaining.slice(lastEnd);
      // Italic: *text*
      const italicRegex = /\*(.+?)\*/g;
      let itLast = 0;
      let itMatch;
      const tailParts = [];
      while ((itMatch = italicRegex.exec(tail)) !== null) {
        if (itMatch.index > itLast) tailParts.push(tail.slice(itLast, itMatch.index));
        tailParts.push(<em key={`i${i}-${idx++}`}>{itMatch[1]}</em>);
        itLast = itMatch.index + itMatch[0].length;
      }
      if (itLast < tail.length) tailParts.push(tail.slice(itLast));
      parts.push(...tailParts);
    }
    // List items: "  - text"
    if (/^\s{2}-\s/.test(line)) {
      return (
        <div key={i} style={{ paddingLeft: 16, marginBottom: 2 }}>
          {"• "}{parts.length > 0 ? parts.map((p,j) => <span key={j}>{typeof p === 'string' ? p.replace(/^\s+-\s/, '') : p}</span>) : line.replace(/^\s+-\s/, '')}
        </div>
      );
    }
    // Empty line → spacer
    if (line.trim() === "") {
      return <div key={i} style={{ height: 8 }} />;
    }
    // Normal line
    return <div key={i}>{parts.length > 0 ? parts : line}</div>;
  });
}

/* ── report summary renderer ────────────────────────────── */
function ReportSummary({ data }) {
  const { summary, file_name, report_length } = data;

  return (
    <div>
      <p style={s.intro}>
        📄 <strong>Medical Report Summary</strong>
      </p>

      {/* Main summary content */}
      <div style={{
        ...s.kbSection,
        borderLeftColor: "#8b5cf6",
        backgroundColor: "#faf5ff",
      }}>
        <div style={{ fontSize: 14, lineHeight: 1.7, color: "#334155" }}>
          {renderMarkdown(summary)}
        </div>
      </div>

      {/* Report stats */}
      {report_length > 0 && (
        <p style={{ fontSize: 12, color: "#9ca3af", marginTop: 10 }}>
          Report processed: {report_length.toLocaleString()} characters
        </p>
      )}

      {/* Warm closing */}
      <hr style={s.sectionSep} />
      <p style={s.intro}>
        If anything is unclear or you'd like to understand a specific section better,
        feel free to ask! 💙
      </p>
    </div>
  );
}

/* ── main component ──────────────────────────────────────── */
export default function ChatMessage({ message }) {
  const isUser = message.role === "user";

  return (
    <div style={s.row(isUser)}>
      {/* Bot avatar */}
      {!isUser && <div style={{ ...s.avatar, ...s.botAvatar }}>M</div>}

      {/* User bubble */}
      {isUser && (
        <div style={s.userBubble}>
          <div>{message.text}</div>
          {message.fileName && (
            <div style={s.fileTag}>📄 {message.fileName}</div>
          )}
        </div>
      )}

      {/* Bot – flowing text (ChatGPT style) */}
      {!isUser && (
        <div style={s.botBody}>
          {message.type === "loading" && (
            <div style={s.typing}>
              <div style={s.dot(0)} />
              <div style={s.dot(0.15)} />
              <div style={s.dot(0.3)} />
            </div>
          )}
          {message.type === "text" && (
            <div style={{ ...s.intro, lineHeight: 1.7 }}>{renderMarkdown(message.text)}</div>
          )}
          {message.type === "prediction" && (
            <ConversationalResponse
              data={message.data}
              uploadedFile={message.uploadedFile}
            />
          )}
          {message.type === "summary" && (
            <ReportSummary data={message.data} />
          )}
          {message.type === "error" && (
            <p style={{ ...s.intro, color: "#ef4444" }}>
              ❌ Oops! {message.text}
            </p>
          )}
        </div>
      )}

      {/* User avatar */}
      {isUser && <div style={{ ...s.avatar, ...s.userAvatar }}>U</div>}
    </div>
  );
}
