/**
 * MedAI – ResultCard Component
 * All predictions displayed inside ONE unified card,
 * each disease section separated by a divider.
 */

import React from "react";

const riskColors = {
  "High Probability": { bg: "#fee2e2", text: "#991b1b", bar: "#ef4444" },
  "Moderate Probability": { bg: "#fef3c7", text: "#92400e", bar: "#f59e0b" },
  "Low Confidence": { bg: "#dbeafe", text: "#1e40af", bar: "#3b82f6" },
};

const styles = {
  container: { marginTop: 24 },
  emergencyBanner: {
    padding: 16,
    backgroundColor: "#fef2f2",
    border: "2px solid #ef4444",
    borderRadius: 10,
    marginBottom: 20,
    textAlign: "center",
  },
  emergencyText: {
    color: "#991b1b",
    fontWeight: 700,
    fontSize: 16,
    margin: 0,
  },
  emergencyKeywords: {
    color: "#b91c1c",
    fontSize: 13,
    marginTop: 6,
  },
  /* --- Single unified card wrapping all predictions --- */
  card: {
    padding: 0,
    borderRadius: 14,
    border: "1px solid #e5e7eb",
    backgroundColor: "#fff",
    boxShadow: "0 2px 8px rgba(0,0,0,0.07)",
    overflow: "hidden",
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: 700,
    color: "#fff",
    margin: 0,
    padding: "16px 24px",
    background: "linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%)",
  },
  /* Each disease section inside the card */
  diseaseSection: {
    padding: "20px 24px",
  },
  divider: {
    height: 1,
    backgroundColor: "#e5e7eb",
    margin: 0,
    border: "none",
  },
  diseaseHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  diseaseName: {
    fontSize: 18,
    fontWeight: 700,
    color: "#1a1a2e",
    textTransform: "capitalize",
  },
  badge: {
    padding: "4px 12px",
    borderRadius: 20,
    fontSize: 12,
    fontWeight: 600,
    whiteSpace: "nowrap",
  },
  progressContainer: {
    height: 8,
    backgroundColor: "#f3f4f6",
    borderRadius: 4,
    overflow: "hidden",
    marginBottom: 6,
  },
  progressBar: {
    height: "100%",
    borderRadius: 4,
    transition: "width 0.6s ease",
  },
  confidenceLabel: {
    fontSize: 13,
    color: "#6b7280",
    marginBottom: 12,
  },
  row: {
    marginBottom: 8,
  },
  rowLabel: {
    fontSize: 13,
    fontWeight: 700,
    color: "#4b5563",
    marginBottom: 2,
    textTransform: "uppercase",
    letterSpacing: "0.4px",
  },
  rowText: {
    fontSize: 14,
    color: "#374151",
    lineHeight: 1.5,
    margin: 0,
  },
  symptomList: {
    display: "flex",
    flexWrap: "wrap",
    gap: 6,
    marginTop: 4,
  },
  symptomChip: {
    padding: "3px 10px",
    backgroundColor: "#e5edff",
    borderRadius: 12,
    fontSize: 12,
    color: "#3730a3",
  },
  precaution: {
    fontSize: 14,
    color: "#059669",
    fontStyle: "italic",
    margin: 0,
    lineHeight: 1.5,
  },
  disclaimer: {
    marginTop: 20,
    padding: 14,
    backgroundColor: "#fffbeb",
    border: "1px solid #fbbf24",
    borderRadius: 8,
    fontSize: 13,
    color: "#92400e",
    textAlign: "center",
  },
};

export default function ResultCard({ data }) {
  if (!data) return null;

  const { predictions, emergency, disclaimer } = data;

  return (
    <div style={styles.container}>
      {/* Emergency Banner */}
      {emergency && emergency.is_emergency && (
        <div style={styles.emergencyBanner}>
          <p style={styles.emergencyText}>{emergency.message}</p>
          <p style={styles.emergencyKeywords}>
            Triggered by: {emergency.triggered_keywords.join(", ")}
          </p>
        </div>
      )}

      {/* ONE unified card containing all predictions */}
      <div style={styles.card}>
        <h2 style={styles.cardTitle}>🔍 Top Predictions</h2>

        {predictions.map((pred, idx) => {
          const risk = riskColors[pred.risk_level] || riskColors["Low Confidence"];
          const info = pred.info || {};

          return (
            <React.Fragment key={`${pred.disease}-${idx}`}>
              {idx > 0 && <hr style={styles.divider} />}

              <div style={styles.diseaseSection}>
                {/* Disease Name + Risk Badge */}
                <div style={styles.diseaseHeader}>
                  <span style={styles.diseaseName}>
                    {idx + 1}. {pred.disease}
                  </span>
                  <span
                    style={{
                      ...styles.badge,
                      backgroundColor: risk.bg,
                      color: risk.text,
                    }}
                  >
                    {pred.risk_level}
                  </span>
                </div>

                {/* Confidence Bar */}
                <div style={styles.progressContainer}>
                  <div
                    style={{
                      ...styles.progressBar,
                      width: `${pred.confidence}%`,
                      backgroundColor: risk.bar,
                    }}
                  />
                </div>
                <div style={styles.confidenceLabel}>
                  Confidence: {pred.confidence}%
                </div>

                {/* Info / Description */}
                {info.description && (
                  <div style={styles.row}>
                    <div style={styles.rowLabel}>Info</div>
                    <p style={styles.rowText}>{info.description}</p>
                  </div>
                )}

                {/* Common Symptoms */}
                {info.common_symptoms && info.common_symptoms.length > 0 && (
                  <div style={styles.row}>
                    <div style={styles.rowLabel}>Symptoms</div>
                    <div style={styles.symptomList}>
                      {info.common_symptoms.map((s) => (
                        <span key={s} style={styles.symptomChip}>{s}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Precaution / Advice */}
                {info.advice && (
                  <div style={{ ...styles.row, marginBottom: 0 }}>
                    <div style={styles.rowLabel}>Precaution</div>
                    <p style={styles.precaution}>💡 {info.advice}</p>
                  </div>
                )}
              </div>
            </React.Fragment>
          );
        })}
      </div>

      {/* Medical Disclaimer */}
      <div style={styles.disclaimer}>
        ⚕️ {disclaimer}
      </div>
    </div>
  );
}
