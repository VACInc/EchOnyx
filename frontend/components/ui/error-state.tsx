import * as React from "react";
import { AlertCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ErrorStateProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "title"> {
  message: React.ReactNode;
  onRetry?: () => void;
  retryLabel?: string;
  title: React.ReactNode;
}

export function ErrorState({ className, message, onRetry, retryLabel = "Retry", title, ...props }: ErrorStateProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-destructive/25 bg-card p-4 text-card-foreground shadow-sm",
        className,
      )}
      role="alert"
      {...props}
    >
      <div className="flex gap-3">
        <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-card-foreground">{title}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{message}</p>
          {onRetry ? (
            <Button type="button" variant="outline" size="sm" className="mt-3" onClick={onRetry}>
              {retryLabel}
            </Button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
