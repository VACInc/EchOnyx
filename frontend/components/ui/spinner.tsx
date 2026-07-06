import { Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";

type SpinnerSize = "sm" | "md" | "lg";

const sizeClasses: Record<SpinnerSize, string> = {
  sm: "h-3.5 w-3.5",
  md: "h-4 w-4",
  lg: "h-6 w-6",
};

export interface SpinnerProps {
  className?: string;
  label?: string;
  size?: SpinnerSize;
}

export function Spinner({ className, label = "Loading", size = "md" }: SpinnerProps) {
  return (
    <Loader2
      className={cn("animate-spin text-current", sizeClasses[size], className)}
      role="status"
      aria-label={label}
    />
  );
}
