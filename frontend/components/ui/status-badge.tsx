import type { LucideIcon } from "lucide-react";
import {
  CheckCircle2,
  Clock,
  Database,
  Download,
  Loader2,
  UploadCloud,
  Wifi,
  WifiOff,
  XCircle,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Badge, type BadgeProps, type BadgeVariant } from "@/components/ui/badge";

type StatusKey =
  | "queued"
  | "processing"
  | "completed"
  | "failed"
  | "uploaded"
  | "loaded"
  | "cached"
  | "uncached"
  | "downloading"
  | "online"
  | "offline";

type StatusConfig = {
  icon: LucideIcon;
  label: string;
  title: string;
  variant: BadgeVariant;
  animated?: boolean;
};

const statusConfig: Record<StatusKey, StatusConfig> = {
  queued: {
    icon: Clock,
    label: "Queued",
    title: "Waiting for processing to start.",
    variant: "warning",
  },
  processing: {
    animated: true,
    icon: Loader2,
    label: "Processing",
    title: "Processing is currently running.",
    variant: "info",
  },
  completed: {
    icon: CheckCircle2,
    label: "Completed",
    title: "Processing completed successfully.",
    variant: "success",
  },
  failed: {
    icon: XCircle,
    label: "Failed",
    title: "Processing failed and needs attention.",
    variant: "destructive",
  },
  uploaded: {
    icon: UploadCloud,
    label: "Uploaded",
    title: "The file uploaded successfully and is ready for the next step.",
    variant: "info",
  },
  loaded: {
    icon: CheckCircle2,
    label: "Loaded",
    title: "The model or resource is loaded and ready.",
    variant: "success",
  },
  cached: {
    icon: Database,
    label: "Cached",
    title: "The model or resource is already cached locally.",
    variant: "success",
  },
  uncached: {
    icon: Database,
    label: "Uncached",
    title: "The model or resource is not cached locally yet.",
    variant: "muted",
  },
  downloading: {
    animated: true,
    icon: Download,
    label: "Downloading",
    title: "The model or resource is downloading now.",
    variant: "info",
  },
  online: {
    icon: Wifi,
    label: "Online",
    title: "The service is reachable and reporting online.",
    variant: "success",
  },
  offline: {
    icon: WifiOff,
    label: "Offline",
    title: "The service is not reachable or is reporting offline.",
    variant: "destructive",
  },
};

export interface StatusBadgeProps extends Omit<BadgeProps, "children" | "variant"> {
  status: string | null | undefined;
}

export function StatusBadge({ className, status, ...props }: StatusBadgeProps) {
  const normalized = status?.toLowerCase() as StatusKey | undefined;
  const config = normalized ? statusConfig[normalized] : undefined;
  const Icon = config?.icon ?? Clock;
  const label = config?.label ?? "Unknown";
  const title = config?.title ?? "Status is not available.";

  return (
    <Badge
      className={cn("capitalize", className)}
      variant={config?.variant ?? "muted"}
      title={title}
      aria-label={`${label}. ${title}`}
      {...props}
    >
      <Icon className={cn("h-3.5 w-3.5", config?.animated && "animate-spin")} aria-hidden="true" />
      {label}
    </Badge>
  );
}
