"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Shield, Loader2, Lock } from "lucide-react";
import { api } from "@/lib/api";

type AuthSession = {
  authenticated: boolean;
  setup_required: boolean;
  actor_label: string | null;
  password_enabled: boolean;
  oidc: {
    enabled: boolean;
    provider_name: string | null;
    login_path: string | null;
  };
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
  oidcProviderName,
  passwordEnabled = true,
}: {
  mode: "login" | "setup";
  onSubmit: (password: string) => Promise<void>;
  pending: boolean;
  oidcProviderName?: string | null;
  passwordEnabled?: boolean;
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
                : passwordEnabled
                  ? "Enter the local admin password."
                  : "Use the configured single sign-on provider."}
            </p>
          </div>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit}>
          {passwordEnabled ? (
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
          ) : null}
          {mode === "setup" && passwordEnabled ? (
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
          {passwordEnabled ? (
            <button
              type="submit"
              disabled={pending}
              className="inline-flex w-full items-center justify-center rounded-2xl bg-blue-500 px-4 py-3 font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:bg-blue-500/60"
            >
              {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : mode === "setup" ? "Create password" : "Sign in"}
            </button>
          ) : null}
          {mode === "login" && oidcProviderName ? (
            <button
              type="button"
              className="inline-flex w-full items-center justify-center rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 font-semibold text-slate-100 transition hover:border-blue-400"
              onClick={() => {
                window.location.href = api.getOidcLoginUrl(window.location.href);
              }}
            >
              Sign in with {oidcProviderName}
            </button>
          ) : null}
        </form>
      </div>
    </div>
  );
}

export function AuthGate({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const [redirectError, setRedirectError] = useState<string | null>(null);
  const sessionQuery = useQuery({
    queryKey: ["authSession"],
    queryFn: api.getAuthSession,
    retry: false,
    staleTime: 0,
  });

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    if (!params.has("auth_error")) {
      return;
    }
    setRedirectError("Single sign-on failed.");
    params.delete("auth_error");
    const next = params.toString();
    const url = `${window.location.pathname}${next ? `?${next}` : ""}${window.location.hash}`;
    window.history.replaceState({}, "", url);
  }, []);

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
      <div className="relative">
        <AuthCard
          mode="login"
          pending={loginMutation.isPending}
          oidcProviderName={sessionQuery.data?.oidc.enabled ? sessionQuery.data.oidc.provider_name : null}
          passwordEnabled={sessionQuery.data?.password_enabled}
          onSubmit={async (password) => {
            await loginMutation.mutateAsync(password);
          }}
        />
        {redirectError ? (
          <div className="pointer-events-none absolute inset-x-0 top-8 flex justify-center px-4">
            <p className="rounded-full border border-rose-400/40 bg-rose-500/10 px-4 py-2 text-sm text-rose-200">
              {redirectError}
            </p>
          </div>
        ) : null}
      </div>
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
