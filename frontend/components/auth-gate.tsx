"use client";

import type { FormEvent, ReactNode } from "react";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Shield, Lock } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";

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
  const passwordInputId = mode === "setup" ? "setup-password" : "login-password";
  const confirmPasswordInputId = "setup-confirm-password";
  const errorId = `${passwordInputId}-error`;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
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
    <div className="dark flex min-h-screen items-center justify-center bg-background px-4 py-8 text-foreground">
      <Card className="w-full max-w-md border-border/80 bg-card/95 p-6 shadow-lg sm:p-8">
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-full bg-info/15 p-3 text-info">
            {mode === "setup" ? <Shield className="h-6 w-6" /> : <Lock className="h-6 w-6" />}
          </div>
          <div>
            <h1 className="text-xl font-semibold text-card-foreground">
              {mode === "setup" ? "Secure EchOnyx" : "Sign in"}
            </h1>
            <p className="text-sm text-muted-foreground">
              {mode === "setup"
                ? "Create the local admin password."
                : passwordEnabled
                  ? "Enter the local admin password."
                  : "Use the configured single sign-on provider."}
            </p>
          </div>
        </div>

        <form className="space-y-4" onSubmit={handleSubmit} aria-describedby={error ? errorId : undefined}>
          {passwordEnabled ? (
            <Field
              id={passwordInputId}
              label="Password"
              description={mode === "setup" ? "Use at least 12 characters." : undefined}
            >
              <Input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete={mode === "setup" ? "new-password" : "current-password"}
                minLength={12}
                required
              />
            </Field>
          ) : null}
          {mode === "setup" && passwordEnabled ? (
            <Field id={confirmPasswordInputId} label="Confirm password">
              <Input
                type="password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                autoComplete="new-password"
                minLength={12}
                required
              />
            </Field>
          ) : null}
          <div
            id={errorId}
            aria-live="polite"
            role={error ? "alert" : undefined}
            className={cn(
              "min-h-5 text-sm font-medium",
              error && "rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-destructive-foreground",
            )}
          >
            {error}
          </div>
          {passwordEnabled ? (
            <Button
              type="submit"
              loading={pending}
              className="w-full"
            >
              {mode === "setup" ? "Create password" : "Sign in"}
            </Button>
          ) : null}
          {mode === "login" && oidcProviderName ? (
            <Button
              type="button"
              variant="outline"
              className="w-full"
              onClick={() => {
                window.location.href = api.getOidcLoginUrl(window.location.href);
              }}
            >
              Sign in with {oidcProviderName}
            </Button>
          ) : null}
        </form>
      </Card>
    </div>
  );
}

export function AuthGate({ children }: { children: ReactNode }) {
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
      <div className="dark flex min-h-screen items-center justify-center bg-background text-foreground">
        <Spinner size="md" />
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
            <p
              className="rounded-full border border-destructive/40 bg-destructive/10 px-4 py-2 text-sm text-destructive-foreground"
              role="alert"
            >
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
