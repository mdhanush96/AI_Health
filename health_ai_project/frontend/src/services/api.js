/**
 * MedAI – API Service Layer
 * Centralized Axios-based API communication.
 * Supports both JSON text and FormData file uploads.
 */

import axios from "axios";

if (!process.env.REACT_APP_API_URL) {
  throw new Error("REACT_APP_API_URL environment variable must be set. See .env.example.");
}
const API_BASE = process.env.REACT_APP_API_URL;

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,  // RAG pipeline can take longer
});

/* ── Response interceptor – handle stale/invalid tokens ──── */
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Clear stale token so subsequent requests go through unauthenticated
      localStorage.removeItem("medai_token");
      localStorage.removeItem("medai_user");
    }
    return Promise.reject(error);
  }
);

/**
 * POST /api/predict-rag/
 * Full pipeline: ClinicalBERT → Symptom Verification → FAISS → T5 Generation
 * @param {string} symptoms – user symptom text
 * @param {File|null} reportFile – optional uploaded medical report
 * @returns {Promise<object>} prediction result with RAG response
 */
export async function predictDisease(symptoms, reportFile = null) {
  if (reportFile) {
    const formData = new FormData();
    if (symptoms) formData.append("symptoms", symptoms);
    formData.append("report", reportFile);
    const response = await api.post("/predict-rag/", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  }
  const response = await api.post("/predict-rag/", { symptoms }, {
    headers: { "Content-Type": "application/json" },
  });
  return response.data;
}

/**
 * GET /api/health/
 */
export async function checkHealth() {
  const response = await api.get("/health/");
  return response.data;
}

/**
 * GET /api/history/
 */
export async function getHistory() {
  const response = await api.get("/history/");
  return response.data;
}

export default api;
