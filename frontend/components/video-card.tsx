"use client";

import { useState } from "react";
import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import { Video, Clock, CheckCircle, Loader2, AlertCircle, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";

interface VideoCardProps {
  video: {
    id: string;
    original_filename: string;
    title: string | null;
    duration_formatted: string;
    status: string;
    created_at: string;
    file_size: number;
    tags?: string[] | null;
  };
  job?: {
    current_step: string | null;
    progress: number;
    step_progress: Record<string, {
      progress?: number;
      step_index?: number;
      step_count?: number;
    }> | null;
  };
}

const STEP_ORDER = [
  "audio_extraction",
  "transcription",
  "diarization",
  "transcript_merge",
  "frame_extraction",
  "vision_analysis",
  "summarization",
  "embedding",
];

const statusConfig = {
  uploaded: {
    icon: Clock,
    color: "text-slate-600 dark:text-slate-200",
    bg: "bg-slate-100 dark:bg-slate-800/70",
    animate: false,
  },
  queued: {
    icon: Clock,
    color: "text-amber-600 dark:text-amber-200",
    bg: "bg-amber-100 dark:bg-amber-500/15",
    animate: false,
  },
  processing: {
    icon: Loader2,
    color: "text-blue-600 dark:text-blue-200",
    bg: "bg-blue-100 dark:bg-blue-500/15",
    animate: true,
  },
  completed: {
    icon: CheckCircle,
    color: "text-emerald-600 dark:text-emerald-200",
    bg: "bg-emerald-100 dark:bg-emerald-500/15",
    animate: false,
  },
  failed: {
    icon: AlertCircle,
    color: "text-rose-600 dark:text-rose-200",
    bg: "bg-rose-100 dark:bg-rose-500/15",
    animate: false,
  },
};

export function VideoCard({ video, job }: VideoCardProps) {
  const [isRetrying, setIsRetrying] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const queryClient = useQueryClient();
  const status = statusConfig[video.status as keyof typeof statusConfig] || statusConfig.uploaded;
  const StatusIcon = status.icon;
  const currentStep = job?.current_step;
  const stepData = currentStep ? job?.step_progress?.[currentStep] : undefined;
  const stepIndex = stepData?.step_index ?? (currentStep ? STEP_ORDER.indexOf(currentStep) + 1 : undefined);
  const stepCount = stepData?.step_count ?? STEP_ORDER.length;
  const progress = job?.progress;
  const statusLabel = video.status === "processing" && stepIndex && progress !== undefined
    ? `Step ${stepIndex}/${stepCount} ${progress.toFixed(0)}%`
    : video.status;
  const showProgress = video.status === "processing" && progress !== undefined;

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  };

  const handleRetry = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsRetrying(true);
    try {
      await api.retryVideo(video.id);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
    } catch (error) {
      console.error("Failed to retry video:", error);
    } finally {
      setIsRetrying(false);
    }
  };

  const handleReset = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResetting(true);
    try {
      await api.resetVideo(video.id);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
    } catch (error) {
      console.error("Failed to reset video:", error);
    } finally {
      setIsResetting(false);
    }
  };

  const canRetry = video.status === "failed";
  const canReset = video.status === "failed" || video.status === "completed";

  return (
    <Link href={`/videos/${video.id}`} className="block">
      <div className="group rounded-2xl border border-slate-200/70 bg-white/70 p-4 transition hover:border-slate-300 hover:bg-white/90 hover:shadow-sm dark:border-slate-700/60 dark:bg-slate-900/60 dark:hover:bg-slate-900">
        <div className="flex flex-col gap-4 md:grid md:grid-cols-[minmax(0,2fr)_150px_120px_120px_140px] md:items-center">
          {/* Title */}
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-300">
              <Video className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <h3 className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
                {video.title || video.original_filename}
              </h3>
              {video.tags && video.tags.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {video.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Status */}
          <div className="flex flex-col gap-2">
            <span className={cn("inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold capitalize", status.bg, status.color)}>
              <StatusIcon className={cn("mr-2 h-3.5 w-3.5", status.animate && "animate-spin")} />
              {statusLabel}
            </span>
            {showProgress && (
              <div className="h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 transition-all"
                  style={{ width: `${progress?.toFixed(0) ?? 0}%` }}
                />
              </div>
            )}
            {(canRetry || canReset) && (
              <div className="flex flex-wrap gap-2">
                {canRetry && (
                  <button
                    onClick={handleRetry}
                    disabled={isRetrying}
                    className="flex items-center rounded-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 px-3 py-1.5 text-xs font-medium text-white transition hover:from-blue-400 hover:via-indigo-400 hover:to-purple-400 disabled:opacity-50"
                  >
                    {isRetrying ? (
                      <Loader2 className="h-3 w-3 animate-spin mr-1" />
                    ) : (
                      <RotateCcw className="h-3 w-3 mr-1" />
                    )}
                    {isRetrying ? "Retrying..." : "Retry"}
                  </button>
                )}
                {canReset && (
                  <button
                    onClick={handleReset}
                    disabled={isResetting}
                    className="flex items-center rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
                  >
                    {isResetting ? (
                      <Loader2 className="h-3 w-3 animate-spin mr-1" />
                    ) : (
                      <RotateCcw className="h-3 w-3 mr-1" />
                    )}
                    {isResetting ? "Resetting..." : "Reset"}
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Duration */}
          <div className="text-sm font-medium text-slate-600 dark:text-slate-300">
            {video.duration_formatted || "Processing..."}
          </div>

          {/* Size */}
          <div className="text-sm font-medium text-slate-600 dark:text-slate-300">
            {formatFileSize(video.file_size)}
          </div>

          {/* Added */}
          <div className="text-sm font-medium text-slate-500 dark:text-slate-400">
            {formatDistanceToNow(new Date(video.created_at), { addSuffix: true })}
          </div>
        </div>
      </div>
    </Link>
  );
}
