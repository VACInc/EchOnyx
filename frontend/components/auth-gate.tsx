"use client";

import { createContext, useContext, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Shield, Loader2, Lock } from "lucide-react";
import { api } from "@/lib/api";

type AuthSession = {
  authenticated: boolean;
  setup_required: boolean;
  actor_label: string | null;
};

type AuthContextValue = {
  session: AuthSession;
  logout: () => Promise<void>;
  refreshSession: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

function AuthCard({
  mode,
  onSubmit,
  pending,
}: {
  mode: "login" | "setup";
  onSubmit: (password: string) => Promise<void>;
  pending: boolean;
}) {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (mode === "setup" && password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    try {
      await onSubmit(password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 px-4">
      <div className="w-full max-w-md rounded-3xl border border-slate-800 bg-slate-900/95 p-8 shadow-2xl">
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-full bg-blue-500/15 p-3 text-blue-300">
            {mode === "setup" ? <Shield className="h-6 w-6" /> : <Lock className="h-6 w-6" />}
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">
              {mode === "setup" ? "Secure EchOnyx" : "Sign in"}
            </h1>
            <p className="text-sm text-slate-400">
              {mode === "setup"
                ? "Create the local admin password."
                : "Enter the local admin password."}
            </p>
          </div>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit}>
          <div>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none transition focus:border-blue-400"
              minLength={12}
              required
            />
          </div>
          {mode === "setup" ? (
            <div>
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                Confirm password
              </label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                className="w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none transition focus:border-blue-400"
                minLength={12}
                required
              />
            </div>
          ) : null}
          {error ? <p className="text-sm text-rose-300">{error}</p> : null}
          <button
            type="submit"
            disabled={pending}
            className="inline-flex w-full items-center justify-center rounded-2xl bg-blue-500 px-4 py-3 font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:bg-blue-500/60"
          >
            {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : mode === "setup" ? "Create password" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}

export function AuthGate({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const sessionQuery = useQuery({
    queryKey: ["authSession"],
    queryFn: api.getAuthSession,
    retry: false,
    staleTime: 0,
  });

  const loginMutation = useMutation({
    mutationFn: api.login,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["authSession"] });
    },
  });

  const setupMutation = useMutation({
    mutationFn: api.setupAuth,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["authSession"] });
    },
  });

  const logoutMutation = useMutation({
    mutationFn: api.logout,
    onSuccess: async () => {
      await queryClient.invalidateQueries();
    },
  });

  const value = useMemo<AuthContextValue | null>(() => {
    if (!sessionQuery.data?.authenticated) {
      return null;
    }
    return {
      session: sessionQuery.data,
      logout: async () => {
        await logoutMutation.mutateAsync();
      },
      refreshSession: async () => {
        await queryClient.invalidateQueries({ queryKey: ["authSession"] });
      },
    };
  }, [logoutMutation, queryClient, sessionQuery.data]);

  if (sessionQuery.isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-200">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }

  if (sessionQuery.data?.setup_required) {
    return (
      <AuthCard
        mode="setup"
        pending={setupMutation.isPending}
        onSubmit={async (password) => {
          await setupMutation.mutateAsync(password);
        }}
      />
    );
  }

  if (!sessionQuery.data?.authenticated) {
    return (
      <AuthCard
        mode="login"
        pending={loginMutation.isPending}
        onSubmit={async (password) => {
          await loginMutation.mutateAsync(password);
        }}
      />
    );
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthGate");
  }
  return context;
}
