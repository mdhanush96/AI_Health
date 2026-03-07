/**
 * MedAI – API Service Layer
 * Centralized Axios-based API communication.
 * Supports both JSON text and FormData file uploads.
 */

import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000/api";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,  // RAG pipeline can take longer
});

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
