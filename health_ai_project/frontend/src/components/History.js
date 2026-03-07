/**
 * MedAI – Prediction History Component
 * Fetches and displays recent prediction logs from backend.
 */

import React, { useEffect, useState } from "react";
import { getHistory } from "../services/api";

const riskColors = {
  "High Probability": "#ef4444",
  "Moderate Probability": "#f59e0b",
  "Low Confidence": "#3b82f6",
};

const styles = {
  container: {
    marginTop: 32,
    padding: 20,
    backgroundColor: "#f8fafc",
    borderRadius: 12,
    border: "1px solid #e2e8f0",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 700,
    color: "#1a1a2e",
    margin: 0,
  },
  refreshBtn: {
    padding: "6px 14px",
    fontSize: 13,
    border: "1px solid #cbd5e1",
    borderRadius: 6,
    backgroundColor: "#fff",
    cursor: "pointer",
    color: "#475569",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 14,
  },
  th: {
    textAlign: "left",
    padding: "10px 12px",
    borderBottom: "2px solid #e2e8f0",
    color: "#64748b",
    fontWeight: 600,
    fontSize: 13,
  },
  td: {
    padding: "10px 12px",
    borderBottom: "1px solid #f1f5f9",
    color: "#334155",
  },
  riskDot: {
    display: "inline-block",
    width: 10,
    height: 10,
    borderRadius: "50%",
    marginRight: 6,
  },
  emptyMsg: {
    textAlign: "center",
    color: "#94a3b8",
    padding: 20,
  },
  errorMsg: {
    textAlign: "center",
    color: "#ef4444",
    padding: 10,
  },
};

export default function History() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchHistory = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await getHistory();
      setLogs(data.history || []);
    } catch {
      setError("Unable to load history. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>📋 Prediction History</h3>
        <button onClick={fetchHistory} disabled={loading} style={styles.refreshBtn}>
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {error && <p style={styles.errorMsg}>{error}</p>}

      {!error && logs.length === 0 && !loading && (
        <p style={styles.emptyMsg}>No predictions yet. Try analyzing some symptoms above.</p>
      )}

      {logs.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>#</th>
                <th style={styles.th}>Disease</th>
                <th style={styles.th}>Confidence</th>
                <th style={styles.th}>Risk</th>
                <th style={styles.th}>Emergency</th>
                <th style={styles.th}>Date</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log, idx) => (
                <tr key={log.id}>
                  <td style={styles.td}>{idx + 1}</td>
                  <td style={{ ...styles.td, fontWeight: 600, textTransform: "capitalize" }}>
                    {log.predicted_disease}
                  </td>
                  <td style={styles.td}>{log.confidence}%</td>
                  <td style={styles.td}>
                    <span
                      style={{
                        ...styles.riskDot,
                        backgroundColor: riskColors[log.risk_level] || "#94a3b8",
                      }}
                    />
                    {log.risk_level}
                  </td>
                  <td style={styles.td}>{log.is_emergency ? "⚠️ Yes" : "No"}</td>
                  <td style={styles.td}>
                    {new Date(log.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
