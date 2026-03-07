/**
 * MedAI – User Profile Page
 * Comprehensive health profile with editable fields and health details.
 */

import React, { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { getProfile, updateProfile } from "../services/auth";

/* ── Styles ──────────────────────────────────────────────── */
const s = {
  page: {
    minHeight: "100vh",
    backgroundColor: "#f0f4f8",
    fontFamily: "'Segoe UI', 'Inter', Arial, sans-serif",
  },
  topBar: {
    background: "linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%)",
    padding: "0 32px",
    height: 60,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    boxShadow: "0 2px 8px rgba(67,97,238,0.25)",
  },
  topBarTitle: {
    fontSize: 20, fontWeight: 800, color: "#fff",
    margin: 0, letterSpacing: "-0.5px",
  },
  backBtn: {
    padding: "8px 18px", border: "1px solid rgba(255,255,255,0.3)",
    borderRadius: 8, backgroundColor: "rgba(255,255,255,0.1)",
    color: "#fff", fontSize: 13, fontWeight: 600, cursor: "pointer",
    transition: "background-color 0.15s", textDecoration: "none",
  },
  container: {
    maxWidth: 900, margin: "0 auto", padding: "30px 24px 60px",
  },
  profileHeader: {
    display: "flex", alignItems: "center", gap: 24,
    backgroundColor: "#fff", borderRadius: 16,
    padding: "28px 32px",
    boxShadow: "0 2px 12px rgba(0,0,0,0.05)",
    marginBottom: 24,
  },
  avatar: {
    width: 80, height: 80, borderRadius: "50%",
    background: "linear-gradient(135deg, #4361ee, #7c3aed)",
    color: "#fff", display: "flex", alignItems: "center",
    justifyContent: "center", fontSize: 32, fontWeight: 700,
    flexShrink: 0, boxShadow: "0 4px 14px rgba(67,97,238,0.3)",
  },
  headerInfo: { flex: 1 },
  headerName: {
    fontSize: 24, fontWeight: 700, color: "#1e293b",
    margin: "0 0 2px",
  },
  headerEmail: { fontSize: 14, color: "#64748b", margin: "0 0 4px" },
  headerMeta: {
    fontSize: 12, color: "#94a3b8", margin: 0,
    display: "flex", alignItems: "center", gap: 12,
  },
  editToggle: {
    padding: "10px 24px", border: "none", borderRadius: 10,
    fontSize: 14, fontWeight: 600, cursor: "pointer",
    transition: "all 0.15s",
  },
  section: {
    backgroundColor: "#fff", borderRadius: 14,
    padding: "24px 28px", marginBottom: 20,
    boxShadow: "0 1px 8px rgba(0,0,0,0.04)",
  },
  sectionTitle: {
    fontSize: 16, fontWeight: 700, color: "#1e293b",
    margin: "0 0 18px", display: "flex", alignItems: "center", gap: 10,
  },
  sectionIcon: {
    width: 30, height: 30, borderRadius: 8,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 16, flexShrink: 0,
  },
  grid: {
    display: "grid", gridTemplateColumns: "1fr 1fr",
    gap: "16px 24px",
  },
  fieldWrap: { marginBottom: 0 },
  fieldLabel: {
    fontSize: 12, fontWeight: 600, color: "#64748b",
    textTransform: "uppercase", letterSpacing: "0.4px",
    marginBottom: 6, display: "block",
  },
  fieldValue: {
    fontSize: 15, color: "#1e293b", fontWeight: 500,
    padding: "10px 14px", backgroundColor: "#f8fafc",
    borderRadius: 8, border: "1px solid #e2e8f0",
    minHeight: 20,
  },
  input: {
    width: "100%", padding: "10px 14px", border: "1.5px solid #d1d5db",
    borderRadius: 8, fontSize: 14, color: "#1e293b",
    backgroundColor: "#fff", outline: "none",
    transition: "border-color 0.2s, box-shadow 0.2s",
    boxSizing: "border-box",
  },
  select: {
    width: "100%", padding: "10px 14px", border: "1.5px solid #d1d5db",
    borderRadius: 8, fontSize: 14, color: "#1e293b",
    backgroundColor: "#fff", outline: "none", cursor: "pointer",
    boxSizing: "border-box",
  },
  textarea: {
    width: "100%", padding: "10px 14px", border: "1.5px solid #d1d5db",
    borderRadius: 8, fontSize: 14, color: "#1e293b",
    backgroundColor: "#fff", outline: "none", minHeight: 70, resize: "vertical",
    fontFamily: "inherit", boxSizing: "border-box",
  },
  fullWidth: { gridColumn: "1 / -1" },
  btnRow: {
    display: "flex", gap: 12, justifyContent: "flex-end",
    marginTop: 24,
  },
  saveBtn: {
    padding: "11px 32px", border: "none", borderRadius: 10,
    fontSize: 14, fontWeight: 700, color: "#fff",
    background: "linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%)",
    cursor: "pointer", boxShadow: "0 4px 14px rgba(67, 97, 238, 0.3)",
    transition: "opacity 0.15s",
  },
  cancelBtn: {
    padding: "11px 24px", border: "1.5px solid #d1d5db", borderRadius: 10,
    fontSize: 14, fontWeight: 600, color: "#475569",
    backgroundColor: "#fff", cursor: "pointer",
    transition: "background-color 0.15s",
  },
  toast: {
    position: "fixed", top: 20, right: 20,
    padding: "14px 24px", borderRadius: 10,
    fontSize: 14, fontWeight: 600, color: "#fff",
    zIndex: 9999, boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
    transition: "opacity 0.3s",
  },
  loading: {
    display: "flex", alignItems: "center", justifyContent: "center",
    height: "60vh", color: "#64748b", fontSize: 16,
  },
};

const GENDERS = [
  { value: "", label: "Not specified" },
  { value: "male", label: "Male" },
  { value: "female", label: "Female" },
  { value: "other", label: "Other" },
];

const BLOOD_GROUPS = [
  { value: "unknown", label: "Not specified" },
  { value: "A+", label: "A+" },
  { value: "A-", label: "A−" },
  { value: "B+", label: "B+" },
  { value: "B-", label: "B−" },
  { value: "AB+", label: "AB+" },
  { value: "AB-", label: "AB−" },
  { value: "O+", label: "O+" },
  { value: "O-", label: "O−" },
];

export default function ProfilePage() {
  const navigate = useNavigate();
  const { user, refreshProfile } = useAuth();

  const [profile, setProfile] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [form, setForm] = useState({});
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState(null);
  const [pageLoading, setPageLoading] = useState(true);

  const showToast = useCallback((msg, type = "success") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  }, []);

  /* Load profile on mount */
  useEffect(() => {
    getProfile()
      .then((data) => {
        setProfile(data);
        setForm(buildFormFromProfile(data));
      })
      .catch(() => showToast("Failed to load profile.", "error"))
      .finally(() => setPageLoading(false));
  }, []);

  function buildFormFromProfile(p) {
    return {
      first_name: p.user?.first_name || "",
      last_name: p.user?.last_name || "",
      email: p.user?.email || "",
      phone: p.phone || "",
      date_of_birth: p.date_of_birth || "",
      gender: p.gender || "",
      blood_group: p.blood_group || "unknown",
      height_cm: p.height_cm || "",
      weight_kg: p.weight_kg || "",
      allergies: p.allergies || "",
      medical_conditions: p.medical_conditions || "",
      emergency_contact: p.emergency_contact || "",
      address: p.address || "",
    };
  }

  const handleChange = (field) => (e) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const handleCancel = () => {
    if (profile) setForm(buildFormFromProfile(profile));
    setEditMode(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const payload = { ...form };
      // Convert numeric fields
      if (payload.height_cm) payload.height_cm = parseFloat(payload.height_cm) || null;
      else payload.height_cm = null;
      if (payload.weight_kg) payload.weight_kg = parseFloat(payload.weight_kg) || null;
      else payload.weight_kg = null;
      if (!payload.date_of_birth) payload.date_of_birth = null;

      const updated = await updateProfile(payload);
      setProfile(updated);
      setForm(buildFormFromProfile(updated));
      await refreshProfile();
      setEditMode(false);
      showToast("Profile updated successfully!", "success");
    } catch (err) {
      const msg =
        err.response?.data?.error ||
        "Failed to save changes. Please try again.";
      showToast(msg, "error");
    } finally {
      setSaving(false);
    }
  };

  const inputFocus = (e) => {
    e.target.style.borderColor = "#4361ee";
    e.target.style.boxShadow = "0 0 0 3px rgba(67,97,238,0.10)";
  };
  const inputBlur = (e) => {
    e.target.style.borderColor = "#d1d5db";
    e.target.style.boxShadow = "none";
  };

  const displayVal = (val) => val || "—";

  if (pageLoading) {
    return (
      <div style={s.page}>
        <div style={s.topBar}>
          <h1 style={s.topBarTitle}>🏥 MedAI</h1>
        </div>
        <div style={s.loading}>Loading profile...</div>
      </div>
    );
  }

  const fullName = [profile?.user?.first_name, profile?.user?.last_name]
    .filter(Boolean).join(" ") || profile?.user?.username || "User";
  const initial = profile?.avatar_initial || fullName[0]?.toUpperCase() || "U";
  const joinDate = profile?.user?.date_joined
    ? new Date(profile.user.date_joined).toLocaleDateString("en-IN", {
        year: "numeric", month: "long", day: "numeric",
      })
    : "";

  /* BMI calculation */
  const bmi =
    profile?.height_cm && profile?.weight_kg
      ? (profile.weight_kg / ((profile.height_cm / 100) ** 2)).toFixed(1)
      : null;

  return (
    <div style={s.page}>
      {/* Top bar */}
      <div style={s.topBar}>
        <h1 style={s.topBarTitle}>🏥 MedAI</h1>
        <button
          style={s.backBtn}
          onClick={() => navigate("/")}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "rgba(255,255,255,0.2)")}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "rgba(255,255,255,0.1)")}
        >
          ← Back to Chat
        </button>
      </div>

      {/* Toast notification */}
      {toast && (
        <div
          style={{
            ...s.toast,
            backgroundColor: toast.type === "success" ? "#22c55e" : "#ef4444",
          }}
        >
          {toast.type === "success" ? "✅" : "❌"} {toast.msg}
        </div>
      )}

      <div style={s.container}>
        {/* Profile header card */}
        <div style={s.profileHeader}>
          <div style={s.avatar}>{initial}</div>
          <div style={s.headerInfo}>
            <h2 style={s.headerName}>{fullName}</h2>
            <p style={s.headerEmail}>{profile?.user?.email}</p>
            <p style={s.headerMeta}>
              <span>@{profile?.user?.username}</span>
              {joinDate && <span>• Member since {joinDate}</span>}
              {bmi && <span>• BMI: {bmi}</span>}
            </p>
          </div>
          <button
            style={{
              ...s.editToggle,
              backgroundColor: editMode ? "#fee2e2" : "#eef2ff",
              color: editMode ? "#dc2626" : "#4361ee",
            }}
            onClick={() => (editMode ? handleCancel() : setEditMode(true))}
          >
            {editMode ? "✕ Cancel" : "✏️ Edit Profile"}
          </button>
        </div>

        {/* Personal Information */}
        <div style={s.section}>
          <h3 style={s.sectionTitle}>
            <div style={{ ...s.sectionIcon, backgroundColor: "#eef2ff", color: "#4361ee" }}>👤</div>
            Personal Information
          </h3>
          <div style={s.grid}>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>First Name</label>
              {editMode ? (
                <input style={s.input} value={form.first_name} onChange={handleChange("first_name")}
                  placeholder="First name" onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>{displayVal(profile?.user?.first_name)}</div>
              )}
            </div>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Last Name</label>
              {editMode ? (
                <input style={s.input} value={form.last_name} onChange={handleChange("last_name")}
                  placeholder="Last name" onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>{displayVal(profile?.user?.last_name)}</div>
              )}
            </div>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Email</label>
              {editMode ? (
                <input style={s.input} type="email" value={form.email} onChange={handleChange("email")}
                  placeholder="email@example.com" onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>{displayVal(profile?.user?.email)}</div>
              )}
            </div>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Phone</label>
              {editMode ? (
                <input style={s.input} type="tel" value={form.phone} onChange={handleChange("phone")}
                  placeholder="+91 98765 43210" onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>{displayVal(profile?.phone)}</div>
              )}
            </div>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Date of Birth</label>
              {editMode ? (
                <input style={s.input} type="date" value={form.date_of_birth}
                  onChange={handleChange("date_of_birth")} onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>
                  {profile?.date_of_birth
                    ? new Date(profile.date_of_birth).toLocaleDateString("en-IN", {
                        year: "numeric", month: "long", day: "numeric",
                      })
                    : "—"}
                </div>
              )}
            </div>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Gender</label>
              {editMode ? (
                <select style={s.select} value={form.gender} onChange={handleChange("gender")}>
                  {GENDERS.map((g) => (
                    <option key={g.value} value={g.value}>{g.label}</option>
                  ))}
                </select>
              ) : (
                <div style={s.fieldValue}>
                  {GENDERS.find((g) => g.value === profile?.gender)?.label || "—"}
                </div>
              )}
            </div>
            <div style={{ ...s.fieldWrap, ...s.fullWidth }}>
              <label style={s.fieldLabel}>Address</label>
              {editMode ? (
                <textarea style={s.textarea} value={form.address} onChange={handleChange("address")}
                  placeholder="Your address" rows={2} onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>{displayVal(profile?.address)}</div>
              )}
            </div>
          </div>
        </div>

        {/* Health Information */}
        <div style={s.section}>
          <h3 style={s.sectionTitle}>
            <div style={{ ...s.sectionIcon, backgroundColor: "#fef3c7", color: "#d97706" }}>🩺</div>
            Health Information
          </h3>
          <div style={s.grid}>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Blood Group</label>
              {editMode ? (
                <select style={s.select} value={form.blood_group} onChange={handleChange("blood_group")}>
                  {BLOOD_GROUPS.map((bg) => (
                    <option key={bg.value} value={bg.value}>{bg.label}</option>
                  ))}
                </select>
              ) : (
                <div style={s.fieldValue}>
                  {BLOOD_GROUPS.find((bg) => bg.value === profile?.blood_group)?.label || "—"}
                </div>
              )}
            </div>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Height (cm)</label>
              {editMode ? (
                <input style={s.input} type="number" step="0.1" value={form.height_cm}
                  onChange={handleChange("height_cm")} placeholder="e.g. 170"
                  onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>
                  {profile?.height_cm ? `${profile.height_cm} cm` : "—"}
                </div>
              )}
            </div>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Weight (kg)</label>
              {editMode ? (
                <input style={s.input} type="number" step="0.1" value={form.weight_kg}
                  onChange={handleChange("weight_kg")} placeholder="e.g. 65"
                  onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>
                  {profile?.weight_kg ? `${profile.weight_kg} kg` : "—"}
                </div>
              )}
            </div>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>BMI</label>
              <div style={{
                ...s.fieldValue,
                backgroundColor: bmi
                  ? parseFloat(bmi) < 18.5 ? "#fef9c3"
                    : parseFloat(bmi) <= 24.9 ? "#dcfce7"
                    : parseFloat(bmi) <= 29.9 ? "#fef3c7"
                    : "#fee2e2"
                  : "#f8fafc",
                color: bmi
                  ? parseFloat(bmi) < 18.5 ? "#a16207"
                    : parseFloat(bmi) <= 24.9 ? "#16a34a"
                    : parseFloat(bmi) <= 29.9 ? "#d97706"
                    : "#dc2626"
                  : "#1e293b",
                fontWeight: bmi ? 600 : 400,
              }}>
                {bmi
                  ? `${bmi} (${
                      parseFloat(bmi) < 18.5 ? "Underweight"
                        : parseFloat(bmi) <= 24.9 ? "Normal"
                        : parseFloat(bmi) <= 29.9 ? "Overweight"
                        : "Obese"
                    })`
                  : "—"}
              </div>
            </div>
            <div style={{ ...s.fieldWrap, ...s.fullWidth }}>
              <label style={s.fieldLabel}>Allergies</label>
              {editMode ? (
                <textarea style={s.textarea} value={form.allergies} onChange={handleChange("allergies")}
                  placeholder="e.g. Peanuts, Penicillin, Dust" rows={2}
                  onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>{displayVal(profile?.allergies)}</div>
              )}
            </div>
            <div style={{ ...s.fieldWrap, ...s.fullWidth }}>
              <label style={s.fieldLabel}>Medical Conditions</label>
              {editMode ? (
                <textarea style={s.textarea} value={form.medical_conditions}
                  onChange={handleChange("medical_conditions")}
                  placeholder="e.g. Diabetes Type 2, Hypertension" rows={2}
                  onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>{displayVal(profile?.medical_conditions)}</div>
              )}
            </div>
          </div>
        </div>

        {/* Emergency Contact */}
        <div style={s.section}>
          <h3 style={s.sectionTitle}>
            <div style={{ ...s.sectionIcon, backgroundColor: "#fce7f3", color: "#db2777" }}>🚨</div>
            Emergency Contact
          </h3>
          <div style={s.grid}>
            <div style={s.fieldWrap}>
              <label style={s.fieldLabel}>Emergency Contact Number</label>
              {editMode ? (
                <input style={s.input} type="tel" value={form.emergency_contact}
                  onChange={handleChange("emergency_contact")}
                  placeholder="+91 98765 43210" onFocus={inputFocus} onBlur={inputBlur} />
              ) : (
                <div style={s.fieldValue}>{displayVal(profile?.emergency_contact)}</div>
              )}
            </div>
          </div>
        </div>

        {/* Save / Cancel buttons */}
        {editMode && (
          <div style={s.btnRow}>
            <button
              style={s.cancelBtn}
              onClick={handleCancel}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f1f5f9")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#fff")}
            >
              Cancel
            </button>
            <button
              style={{ ...s.saveBtn, opacity: saving ? 0.6 : 1 }}
              disabled={saving}
              onClick={handleSave}
            >
              {saving ? "Saving..." : "💾 Save Changes"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
