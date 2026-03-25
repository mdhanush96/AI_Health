/**
 * MedAI – Auth Service Layer
 * Handles registration, login, logout, and profile API calls.
 */

import api from "./api";

const TOKEN_KEY = "medai_token";
const USER_KEY = "medai_user";

/* ── Token helpers ──────────────────────────────────────── */

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function removeToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function getSavedUser() {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function saveUser(user) {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

/* ── Axios interceptor – attach token to every request ──── */

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Token ${token}`;
  }
  return config;
});

/* ── API calls ──────────────────────────────────────────── */

export async function registerUser({ username, email, password, firstName, lastName }) {
  try {
    const response = await api.post("/auth/register/", {
      username,
      email,
      password,
      first_name: firstName || "",
      last_name: lastName || "",
    });
    const { token, user, profile } = response.data;
    setToken(token);
    saveUser({ ...user, ...profile });
    return response.data;
  } catch (error) {
    throw error;
  }
}

export async function loginUser({ username, password }) {
  try {
    const response = await api.post("/auth/login/", { username, password });
    const { token, user, profile } = response.data;
    setToken(token);
    saveUser({ ...user, ...profile });
    return response.data;
  } catch (error) {
    throw error;
  }
}

export async function logoutUser() {
  try {
    await api.post("/auth/logout/");
  } catch {
    /* even if request fails, clear local state */
  }
  removeToken();
}

export async function getProfile() {
  const response = await api.get("/auth/profile/");
  return response.data;
}

export async function updateProfile(data) {
  const response = await api.put("/auth/profile/", data);
  // Update saved user with new info
  const saved = getSavedUser() || {};
  saveUser({
    ...saved,
    ...data,
    avatar_initial: response.data.avatar_initial || saved.avatar_initial,
  });
  return response.data;
}
