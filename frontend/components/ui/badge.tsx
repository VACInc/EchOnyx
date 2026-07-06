import * as React from "react";

import { cn } from "@/lib/utils";

export type BadgeVariant =
  | "default"
  | "secondary"
  | "success"
  | "warning"
  | "info"
  | "destructive"
  | "muted"
  | "outline";

const badgeVariants: Record<BadgeVariant, string> = {
  default: "border-primary/15 bg-primary text-primary-foreground",
  secondary: "border-secondary bg-secondary text-secondary-foreground",
  success: "border-success/20 bg-success text-success-foreground",
  warning: "border-warning/25 bg-warning text-warning-foreground",
  info: "border-info/20 bg-info text-info-foreground",
  destructive: "border-destructive/20 bg-destructive text-destructive-foreground",
  muted: "border-muted bg-muted text-muted-foreground",
  outline: "border-border bg-transparent text-foreground",
};

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
}

export function Badge({ className, variant = "secondary", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex min-h-6 items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        badgeVariants[variant],
        className,
      )}
      {...props}
    />
  );
}
