/**
 * MedAI – Login Page
 * Professional login form with modern medical-themed design.
 */

import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

/* ── Styles ──────────────────────────────────────────────── */
const s = {
  page: {
    minHeight: "100vh",
    display: "flex",
    fontFamily: "'Segoe UI', 'Inter', Arial, sans-serif",
    backgroundColor: "#f0f4f8",
  },
  /* Left branding panel */
  brandPanel: {
    flex: "0 0 45%",
    background: "linear-gradient(135deg, #4361ee 0%, #3a0ca3 60%, #2d0a6e 100%)",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    padding: "60px 40px",
    color: "#fff",
    position: "relative",
    overflow: "hidden",
  },
  brandBg: {
    position: "absolute",
    top: -80,
    right: -80,
    width: 320,
    height: 320,
    borderRadius: "50%",
    background: "rgba(255,255,255, 0.06)",
  },
  brandBg2: {
    position: "absolute",
    bottom: -60,
    left: -60,
    width: 240,
    height: 240,
    borderRadius: "50%",
    background: "rgba(255,255,255, 0.04)",
  },
  brandIcon: {
    fontSize: 64,
    marginBottom: 20,
    position: "relative",
    zIndex: 1,
  },
  brandTitle: {
    fontSize: 36,
    fontWeight: 800,
    margin: "0 0 12px",
    letterSpacing: "-1px",
    position: "relative",
    zIndex: 1,
  },
  brandSub: {
    fontSize: 16,
    opacity: 0.85,
    maxWidth: 340,
    textAlign: "center",
    lineHeight: 1.6,
    position: "relative",
    zIndex: 1,
  },
  features: {
    marginTop: 40,
    display: "flex",
    flexDirection: "column",
    gap: 14,
    position: "relative",
    zIndex: 1,
  },
  featureItem: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    fontSize: 14,
    opacity: 0.9,
  },
  featureIcon: {
    width: 28,
    height: 28,
    borderRadius: "50%",
    background: "rgba(255,255,255, 0.15)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 14,
    flexShrink: 0,
  },
  /* Right form panel */
  formPanel: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    padding: "40px 24px",
  },
  formCard: {
    width: "100%",
    maxWidth: 420,
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: "44px 36px",
    boxShadow: "0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04)",
  },
  formTitle: {
    fontSize: 26,
    fontWeight: 800,
    color: "#1e293b",
    margin: "0 0 4px",
  },
  formSub: {
    fontSize: 14,
    color: "#64748b",
    margin: "0 0 28px",
  },
  label: {
    display: "block",
    fontSize: 13,
    fontWeight: 600,
    color: "#374151",
    marginBottom: 6,
  },
  inputWrap: {
    position: "relative",
    marginBottom: 20,
  },
  input: {
    width: "100%",
    padding: "12px 14px 12px 42px",
    border: "1.5px solid #d1d5db",
    borderRadius: 10,
    fontSize: 14,
    color: "#1e293b",
    backgroundColor: "#f8fafc",
    outline: "none",
    transition: "border-color 0.2s, box-shadow 0.2s",
    boxSizing: "border-box",
  },
  inputIcon: {
    position: "absolute",
    left: 14,
    top: "50%",
    transform: "translateY(-50%)",
    fontSize: 16,
    color: "#94a3b8",
    pointerEvents: "none",
  },
  forgotLink: {
    display: "block",
    textAlign: "right",
    fontSize: 13,
    color: "#4361ee",
    textDecoration: "none",
    marginTop: -12,
    marginBottom: 20,
    fontWeight: 500,
  },
  btn: {
    width: "100%",
    padding: "13px",
    border: "none",
    borderRadius: 10,
    fontSize: 15,
    fontWeight: 700,
    color: "#fff",
    background: "linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%)",
    cursor: "pointer",
    transition: "opacity 0.2s, transform 0.1s",
    boxShadow: "0 4px 14px rgba(67, 97, 238, 0.35)",
  },
  btnDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  divider: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    margin: "24px 0",
    color: "#94a3b8",
    fontSize: 12,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: "#e2e8f0",
  },
  registerLink: {
    textAlign: "center",
    fontSize: 14,
    color: "#64748b",
    marginTop: 0,
  },
  link: {
    color: "#4361ee",
    fontWeight: 600,
    textDecoration: "none",
  },
  error: {
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: 8,
    padding: "10px 14px",
    fontSize: 13,
    color: "#dc2626",
    marginBottom: 18,
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
};

export default function LoginPage() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [form, setForm] = useState({ username: "", password: "" });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleChange = (field) => (e) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.username.trim() || !form.password) {
      setError("Please fill in all fields.");
      return;
    }
    setSubmitting(true);
    try {
      await login({ username: form.username.trim(), password: form.password });
      navigate("/", { replace: true });
    } catch (err) {
      const msg =
        err.response?.data?.error ||
        err.response?.data?.details?.username?.[0] ||
        "Login failed. Please check your credentials.";
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={s.page}>
      {/* Left branding */}
      <div style={s.brandPanel}>
        <div style={s.brandBg} />
        <div style={s.brandBg2} />
        <div style={s.brandIcon}>🏥</div>
        <h1 style={s.brandTitle}>MedAI</h1>
        <p style={s.brandSub}>
          Your AI-powered health companion. Get instant symptom analysis and
          personalized medical guidance.
        </p>
        <div style={s.features}>
          <div style={s.featureItem}>
            <div style={s.featureIcon}>🧠</div>
            <span>ClinicalBERT Disease Prediction</span>
          </div>
          <div style={s.featureItem}>
            <div style={s.featureIcon}>📋</div>
            <span>Medical Report Summarization</span>
          </div>
          <div style={s.featureItem}>
            <div style={s.featureIcon}>🔒</div>
            <span>Secure & Private Health Data</span>
          </div>
          <div style={s.featureItem}>
            <div style={s.featureIcon}>💊</div>
            <span>Personalized Health Insights</span>
          </div>
        </div>
      </div>

      {/* Right form */}
      <div style={s.formPanel}>
        <div style={s.formCard}>
          <h2 style={s.formTitle}>Welcome back</h2>
          <p style={s.formSub}>Sign in to continue to MedAI</p>

          {error && (
            <div style={s.error}>
              <span>⚠️</span> {error}
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div style={s.inputWrap}>
              <label style={s.label}>Username</label>
              <span style={s.inputIcon}>👤</span>
              <input
                style={s.input}
                type="text"
                placeholder="Enter your username"
                value={form.username}
                onChange={handleChange("username")}
                autoComplete="username"
                autoFocus
                onFocus={(e) => {
                  e.target.style.borderColor = "#4361ee";
                  e.target.style.boxShadow = "0 0 0 3px rgba(67,97,238,0.12)";
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = "#d1d5db";
                  e.target.style.boxShadow = "none";
                }}
              />
            </div>

            <div style={s.inputWrap}>
              <label style={s.label}>Password</label>
              <span style={s.inputIcon}>🔒</span>
              <input
                style={s.input}
                type="password"
                placeholder="Enter your password"
                value={form.password}
                onChange={handleChange("password")}
                autoComplete="current-password"
                onFocus={(e) => {
                  e.target.style.borderColor = "#4361ee";
                  e.target.style.boxShadow = "0 0 0 3px rgba(67,97,238,0.12)";
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = "#d1d5db";
                  e.target.style.boxShadow = "none";
                }}
              />
            </div>

            <button
              type="submit"
              disabled={submitting}
              style={{ ...s.btn, ...(submitting ? s.btnDisabled : {}) }}
              onMouseDown={(e) => {
                if (!submitting) e.currentTarget.style.transform = "scale(0.98)";
              }}
              onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            >
              {submitting ? "Signing in..." : "Sign In"}
            </button>
          </form>

          <div style={s.divider}>
            <div style={s.dividerLine} />
            <span>OR</span>
            <div style={s.dividerLine} />
          </div>

          <p style={s.registerLink}>
            Don't have an account?{" "}
            <Link to="/register" style={s.link}>
              Create one now
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
