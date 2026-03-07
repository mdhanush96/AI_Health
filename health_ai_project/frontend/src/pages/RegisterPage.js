/**
 * MedAI – Registration Page
 * Professional sign-up form with real-time validation.
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
  brandIcon: { fontSize: 64, marginBottom: 20, position: "relative", zIndex: 1 },
  brandTitle: {
    fontSize: 36, fontWeight: 800, margin: "0 0 12px",
    letterSpacing: "-1px", position: "relative", zIndex: 1,
  },
  brandSub: {
    fontSize: 16, opacity: 0.85, maxWidth: 340, textAlign: "center",
    lineHeight: 1.6, position: "relative", zIndex: 1,
  },
  steps: {
    marginTop: 40, display: "flex", flexDirection: "column", gap: 18,
    position: "relative", zIndex: 1,
  },
  step: {
    display: "flex", alignItems: "center", gap: 14,
  },
  stepNum: {
    width: 32, height: 32, borderRadius: "50%",
    background: "rgba(255,255,255,0.18)",
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 14, fontWeight: 700, flexShrink: 0,
  },
  stepText: { fontSize: 14, opacity: 0.9 },

  formPanel: {
    flex: 1, display: "flex", flexDirection: "column",
    justifyContent: "center", alignItems: "center", padding: "40px 24px",
    overflowY: "auto",
  },
  formCard: {
    width: "100%", maxWidth: 440, backgroundColor: "#fff",
    borderRadius: 16, padding: "36px 36px 30px",
    boxShadow: "0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04)",
  },
  formTitle: { fontSize: 26, fontWeight: 800, color: "#1e293b", margin: "0 0 4px" },
  formSub: { fontSize: 14, color: "#64748b", margin: "0 0 24px" },
  row: { display: "flex", gap: 14 },
  inputWrap: { position: "relative", marginBottom: 18, flex: 1 },
  label: { display: "block", fontSize: 13, fontWeight: 600, color: "#374151", marginBottom: 6 },
  input: {
    width: "100%", padding: "11px 14px 11px 40px",
    border: "1.5px solid #d1d5db", borderRadius: 10, fontSize: 14,
    color: "#1e293b", backgroundColor: "#f8fafc", outline: "none",
    transition: "border-color 0.2s, box-shadow 0.2s", boxSizing: "border-box",
  },
  inputIcon: {
    position: "absolute", left: 13, top: "50%",
    transform: "translateY(-50%)", fontSize: 15, color: "#94a3b8",
    pointerEvents: "none",
  },
  hint: { fontSize: 11, color: "#94a3b8", marginTop: 4, display: "block" },
  strengthBar: {
    height: 4, borderRadius: 4, marginTop: 6, transition: "all 0.3s",
  },
  btn: {
    width: "100%", padding: "13px", border: "none", borderRadius: 10,
    fontSize: 15, fontWeight: 700, color: "#fff",
    background: "linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%)",
    cursor: "pointer", transition: "opacity 0.2s, transform 0.1s",
    boxShadow: "0 4px 14px rgba(67, 97, 238, 0.35)", marginTop: 4,
  },
  btnDisabled: { opacity: 0.6, cursor: "not-allowed" },
  divider: {
    display: "flex", alignItems: "center", gap: 12,
    margin: "22px 0 18px", color: "#94a3b8", fontSize: 12,
  },
  dividerLine: { flex: 1, height: 1, backgroundColor: "#e2e8f0" },
  loginLink: { textAlign: "center", fontSize: 14, color: "#64748b", margin: 0 },
  link: { color: "#4361ee", fontWeight: 600, textDecoration: "none" },
  error: {
    backgroundColor: "#fef2f2", border: "1px solid #fecaca",
    borderRadius: 8, padding: "10px 14px", fontSize: 13,
    color: "#dc2626", marginBottom: 16, display: "flex", alignItems: "center", gap: 8,
  },
  success: {
    backgroundColor: "#f0fdf4", border: "1px solid #bbf7d0",
    borderRadius: 8, padding: "10px 14px", fontSize: 13,
    color: "#16a34a", marginBottom: 16, display: "flex", alignItems: "center", gap: 8,
  },
};

/* ── Password strength helper ────────────────────────────── */
function getPasswordStrength(pw) {
  if (!pw) return { level: 0, label: "", color: "#e2e8f0" };
  let score = 0;
  if (pw.length >= 6) score++;
  if (pw.length >= 10) score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;

  if (score <= 1) return { level: 1, label: "Weak", color: "#ef4444" };
  if (score <= 2) return { level: 2, label: "Fair", color: "#f59e0b" };
  if (score <= 3) return { level: 3, label: "Good", color: "#3b82f6" };
  return { level: 4, label: "Strong", color: "#22c55e" };
}

export default function RegisterPage() {
  const navigate = useNavigate();
  const { register } = useAuth();

  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleChange = (field) => (e) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
    setError("");
  };

  const pwStrength = getPasswordStrength(form.password);

  const inputFocus = (e) => {
    e.target.style.borderColor = "#4361ee";
    e.target.style.boxShadow = "0 0 0 3px rgba(67,97,238,0.12)";
  };
  const inputBlur = (e) => {
    e.target.style.borderColor = "#d1d5db";
    e.target.style.boxShadow = "none";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const { firstName, lastName, username, email, password, confirmPassword } = form;

    if (!username.trim() || !email.trim() || !password) {
      setError("Please fill in all required fields.");
      return;
    }
    if (username.trim().length < 3) {
      setError("Username must be at least 3 characters.");
      return;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("Please enter a valid email address.");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setSubmitting(true);
    try {
      await register({
        username: username.trim(),
        email: email.trim(),
        password,
        firstName: firstName.trim(),
        lastName: lastName.trim(),
      });
      navigate("/", { replace: true });
    } catch (err) {
      const data = err.response?.data;
      const msg =
        data?.error ||
        data?.details?.username?.[0] ||
        data?.details?.email?.[0] ||
        data?.details?.password?.[0] ||
        "Registration failed. Please try again.";
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
          Join thousands of users managing their health smarter with AI-powered insights.
        </p>
        <div style={s.steps}>
          <div style={s.step}>
            <div style={s.stepNum}>1</div>
            <span style={s.stepText}>Create your free account</span>
          </div>
          <div style={s.step}>
            <div style={s.stepNum}>2</div>
            <span style={s.stepText}>Set up your health profile</span>
          </div>
          <div style={s.step}>
            <div style={s.stepNum}>3</div>
            <span style={s.stepText}>Get personalized health analysis</span>
          </div>
        </div>
      </div>

      {/* Right form */}
      <div style={s.formPanel}>
        <div style={s.formCard}>
          <h2 style={s.formTitle}>Create Account</h2>
          <p style={s.formSub}>Start your health journey with MedAI</p>

          {error && (
            <div style={s.error}>
              <span>⚠️</span> {error}
            </div>
          )}

          <form onSubmit={handleSubmit}>
            {/* Name row */}
            <div style={s.row}>
              <div style={s.inputWrap}>
                <label style={s.label}>First Name</label>
                <span style={s.inputIcon}>👤</span>
                <input
                  style={s.input}
                  type="text"
                  placeholder="First name"
                  value={form.firstName}
                  onChange={handleChange("firstName")}
                  onFocus={inputFocus}
                  onBlur={inputBlur}
                />
              </div>
              <div style={s.inputWrap}>
                <label style={s.label}>Last Name</label>
                <span style={s.inputIcon}>👤</span>
                <input
                  style={s.input}
                  type="text"
                  placeholder="Last name"
                  value={form.lastName}
                  onChange={handleChange("lastName")}
                  onFocus={inputFocus}
                  onBlur={inputBlur}
                />
              </div>
            </div>

            {/* Username */}
            <div style={s.inputWrap}>
              <label style={s.label}>Username *</label>
              <span style={s.inputIcon}>@</span>
              <input
                style={s.input}
                type="text"
                placeholder="Choose a username"
                value={form.username}
                onChange={handleChange("username")}
                autoComplete="username"
                onFocus={inputFocus}
                onBlur={inputBlur}
              />
            </div>

            {/* Email */}
            <div style={s.inputWrap}>
              <label style={s.label}>Email *</label>
              <span style={s.inputIcon}>📧</span>
              <input
                style={s.input}
                type="email"
                placeholder="you@example.com"
                value={form.email}
                onChange={handleChange("email")}
                autoComplete="email"
                onFocus={inputFocus}
                onBlur={inputBlur}
              />
            </div>

            {/* Password */}
            <div style={s.inputWrap}>
              <label style={s.label}>Password *</label>
              <span style={s.inputIcon}>🔒</span>
              <input
                style={s.input}
                type="password"
                placeholder="Min. 6 characters"
                value={form.password}
                onChange={handleChange("password")}
                autoComplete="new-password"
                onFocus={inputFocus}
                onBlur={inputBlur}
              />
              {form.password && (
                <>
                  <div
                    style={{
                      ...s.strengthBar,
                      width: `${pwStrength.level * 25}%`,
                      backgroundColor: pwStrength.color,
                    }}
                  />
                  <span style={{ ...s.hint, color: pwStrength.color }}>
                    {pwStrength.label}
                  </span>
                </>
              )}
            </div>

            {/* Confirm Password */}
            <div style={s.inputWrap}>
              <label style={s.label}>Confirm Password *</label>
              <span style={s.inputIcon}>🔒</span>
              <input
                style={{
                  ...s.input,
                  borderColor:
                    form.confirmPassword && form.confirmPassword !== form.password
                      ? "#ef4444"
                      : "#d1d5db",
                }}
                type="password"
                placeholder="Re-enter your password"
                value={form.confirmPassword}
                onChange={handleChange("confirmPassword")}
                autoComplete="new-password"
                onFocus={inputFocus}
                onBlur={inputBlur}
              />
              {form.confirmPassword && form.confirmPassword !== form.password && (
                <span style={{ ...s.hint, color: "#ef4444" }}>Passwords don't match</span>
              )}
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
              {submitting ? "Creating account..." : "Create Account"}
            </button>
          </form>

          <div style={s.divider}>
            <div style={s.dividerLine} />
            <span>OR</span>
            <div style={s.dividerLine} />
          </div>

          <p style={s.loginLink}>
            Already have an account?{" "}
            <Link to="/login" style={s.link}>
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
