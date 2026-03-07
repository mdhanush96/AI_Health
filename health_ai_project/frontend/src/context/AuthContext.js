/**
 * MedAI – Auth Context
 * Provides authentication state across the application.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import {
  getToken,
  getSavedUser,
  loginUser,
  logoutUser,
  registerUser,
  getProfile,
  removeToken,
} from "../services/auth";

const AuthContext = createContext(null);

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

export default function AuthProvider({ children }) {
  const [user, setUser] = useState(() => getSavedUser());
  const [token, setTokenState] = useState(() => getToken());
  const [loading, setLoading] = useState(false);

  const isAuthenticated = !!token;

  /* On mount, if token exists, fetch latest profile */
  useEffect(() => {
    if (token && !user?.id) {
      setLoading(true);
      getProfile()
        .then((profile) => {
          setUser({
            id: profile.user.id,
            username: profile.user.username,
            email: profile.user.email,
            first_name: profile.user.first_name,
            last_name: profile.user.last_name,
            avatar_initial: profile.avatar_initial,
          });
        })
        .catch(() => {
          removeToken();
          setTokenState(null);
          setUser(null);
        })
        .finally(() => setLoading(false));
    }
  }, []);

  const login = useCallback(async (credentials) => {
    const data = await loginUser(credentials);
    setTokenState(data.token);
    setUser({ ...data.user, ...data.profile });
    return data;
  }, []);

  const register = useCallback(async (info) => {
    const data = await registerUser(info);
    setTokenState(data.token);
    setUser({ ...data.user, ...data.profile });
    return data;
  }, []);

  const logout = useCallback(async () => {
    await logoutUser();
    setTokenState(null);
    setUser(null);
  }, []);

  const refreshProfile = useCallback(async () => {
    const profile = await getProfile();
    const u = {
      id: profile.user.id,
      username: profile.user.username,
      email: profile.user.email,
      first_name: profile.user.first_name,
      last_name: profile.user.last_name,
      avatar_initial: profile.avatar_initial,
      phone: profile.phone,
      gender: profile.gender,
      blood_group: profile.blood_group,
    };
    setUser(u);
    return profile;
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated,
        loading,
        login,
        register,
        logout,
        refreshProfile,
        setUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
