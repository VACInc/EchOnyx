"use client";

import * as React from "react";
import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type ToastVariant = "success" | "error" | "info";

export interface ToastOptions {
  description?: React.ReactNode;
  durationMs?: number;
  title?: React.ReactNode;
  variant: ToastVariant;
}

type ToastItem = ToastOptions & {
  id: string;
};

const variantStyles: Record<ToastVariant, { icon: typeof CheckCircle2; className: string }> = {
  success: {
    icon: CheckCircle2,
    className: "border-success/25 bg-card text-card-foreground",
  },
  error: {
    icon: AlertCircle,
    className: "border-destructive/25 bg-card text-card-foreground",
  },
  info: {
    icon: Info,
    className: "border-info/25 bg-card text-card-foreground",
  },
};

const iconStyles: Record<ToastVariant, string> = {
  success: "text-success",
  error: "text-destructive",
  info: "text-info",
};

const ToastContext = React.createContext<((options: ToastOptions) => string) | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<ToastItem[]>([]);

  const dismiss = React.useCallback((id: string) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const toast = React.useCallback(
    (options: ToastOptions) => {
      const id = crypto.randomUUID();
      const durationMs = options.durationMs ?? (options.variant === "error" ? 8000 : 4000);
      setToasts((current) => [...current, { ...options, id }]);
      window.setTimeout(() => dismiss(id), durationMs);
      return id;
    },
    [dismiss],
  );

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <div
        className="pointer-events-none fixed right-4 top-4 z-[60] flex w-[calc(100vw-2rem)] max-w-sm flex-col gap-3"
        aria-live="polite"
        aria-atomic="false"
      >
        {toasts.map((item) => {
          const styles = variantStyles[item.variant];
          const Icon = styles.icon;
          return (
            <div
              key={item.id}
              className={cn(
                "pointer-events-auto flex items-start gap-3 rounded-lg border p-4 shadow-lg",
                "motion-safe:animate-fade-up",
                styles.className,
              )}
              role="status"
            >
              <Icon className={cn("mt-0.5 h-5 w-5 shrink-0", iconStyles[item.variant])} aria-hidden="true" />
              <div className="min-w-0 flex-1">
                {item.title ? <div className="text-sm font-semibold text-card-foreground">{item.title}</div> : null}
                {item.description ? (
                  <div className={cn("text-sm text-muted-foreground", item.title ? "mt-1" : "")}>
                    {item.description}
                  </div>
                ) : null}
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() => dismiss(item.id)}
                aria-label="Dismiss notification"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = React.useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within ToastProvider");
  }
  return context;
}
