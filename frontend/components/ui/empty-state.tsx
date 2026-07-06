import * as React from "react";

import { cn } from "@/lib/utils";

export interface EmptyStateProps extends React.HTMLAttributes<HTMLDivElement> {
  action?: React.ReactNode;
  headline: React.ReactNode;
  hint: React.ReactNode;
  icon: React.ReactNode;
}

export function EmptyState({ action, className, headline, hint, icon, ...props }: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-card px-6 py-10 text-center text-card-foreground",
        className,
      )}
      {...props}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted text-muted-foreground">{icon}</div>
      <h2 className="mt-4 text-lg font-semibold text-card-foreground">{headline}</h2>
      <p className="mt-2 max-w-md text-sm text-muted-foreground">{hint}</p>
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  );
}
