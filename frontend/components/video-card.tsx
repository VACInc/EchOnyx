"use client";

import { useState } from "react";
import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import { RotateCcw, XCircle, Video } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { cn, formatFileSize } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { StatusBadge } from "@/components/ui/status-badge";
import { Tooltip } from "@/components/ui/tooltip";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { useToast } from "@/components/ui/toast";

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
    duplicate_info?: {
      classification: string | null;
      score: number | null;
      suppressed: boolean | null;
      duplicate_of: { id: string; title: string | null } | null;
    } | null;
  };
  job?: {
    id: string;
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

function duplicateLabel(classification: string): string {
  return classification
    .split("_")
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}

export function VideoCard({ video, job }: VideoCardProps) {
  const [isRetrying, setIsRetrying] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const queryClient = useQueryClient();
  const confirm = useConfirm();
  const toast = useToast();

  const currentStep = job?.current_step;
  const stepData = currentStep ? job?.step_progress?.[currentStep] : undefined;
  const stepIndex = stepData?.step_index ?? (currentStep ? STEP_ORDER.indexOf(currentStep) + 1 : undefined);
  const stepCount = stepData?.step_count ?? STEP_ORDER.length;
  const progress = job?.progress;
  const statusLabel = video.status === "processing" && stepIndex && progress !== undefined
    ? `Step ${stepIndex}/${stepCount} ${progress.toFixed(0)}%`
    : null;
  const showProgress = video.status === "processing" && progress !== undefined;
  const duplicateInfo = video.duplicate_info;
  const isDuplicate = duplicateInfo?.classification?.includes("duplicate") ?? false;

  const handleRetry = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsRetrying(true);
    try {
      await api.retryVideo(video.id);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to retry video";
      toast({
        title: "Retry failed",
        description: message,
        variant: "error",
      });
    } finally {
      setIsRetrying(false);
    }
  };

  const handleReset = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const isCompletedReset = video.status === "completed";
    if (isCompletedReset) {
      const confirmed = await confirm({
        title: "Rerun completed video?",
        description: "This video already completed. Rerun it anyway?",
        confirmLabel: "Rerun",
        cancelLabel: "Keep current result",
        destructive: true,
      });
      if (!confirmed) {
        return;
      }
    }
    setIsResetting(true);
    try {
      await api.resetVideo(video.id, { force: isCompletedReset });
      queryClient.invalidateQueries({ queryKey: ["videos"] });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to reset video";
      toast({
        title: "Reset failed",
        description: message,
        variant: "error",
      });
    } finally {
      setIsResetting(false);
    }
  };

  const handleCancel = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!job?.id || isCancelling) return;

    const confirmed = await confirm({
      title: "Cancel processing job?",
      description: "This stops the queued or active processing job for this video.",
      confirmLabel: "Cancel job",
      cancelLabel: "Keep running",
      destructive: true,
    });
    if (!confirmed) return;

    setIsCancelling(true);
    try {
      await api.cancelJob(job.id);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["videos"] }),
        queryClient.invalidateQueries({ queryKey: ["jobs"] }),
      ]);
      toast({
        title: "Job cancelled",
        description: "The video processing job was cancelled.",
        variant: "success",
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to cancel job";
      toast({
        title: "Cancel failed",
        description: message,
        variant: "error",
      });
    } finally {
      setIsCancelling(false);
    }
  };

  const canRetry = video.status === "failed";
  const canReset = video.status === "failed" || video.status === "completed";
  const canCancel = (video.status === "queued" || video.status === "processing") && Boolean(job?.id);

  return (
    <Link href={`/videos/${video.id}`} className="block rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background">
      <div className="group rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm transition duration-200 hover:bg-accent/50 hover:shadow-sm">
        <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_150px_120px_120px_140px] md:items-center">
          <div className="flex min-w-0 items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
              <Video className="h-5 w-5" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <h3 className="truncate text-sm font-semibold text-card-foreground">
                  {video.title || video.original_filename}
                </h3>
                {isDuplicate && duplicateInfo?.classification ? (
                  <Tooltip
                    content={
                      <span>
                        {duplicateLabel(duplicateInfo.classification)}
                        {duplicateInfo.score !== null && duplicateInfo.score !== undefined
                          ? `, score ${(duplicateInfo.score * 100).toFixed(1)}%`
                          : ""}
                        {duplicateInfo.duplicate_of?.title
                          ? `. Matches ${duplicateInfo.duplicate_of.title}.`
                          : "."}
                      </span>
                    }
                  >
                    <Badge variant="warning" tabIndex={0}>
                      Duplicate
                    </Badge>
                  </Tooltip>
                ) : null}
              </div>
              {video.tags && video.tags.length > 0 ? (
                <div className="mt-2 flex flex-wrap gap-2">
                  {video.tags.map((tag) => (
                    <Badge key={tag} variant="muted" className="min-h-5 px-2 py-0 text-[11px]">
                      {tag}
                    </Badge>
                  ))}
                </div>
              ) : null}
            </div>
          </div>

          <div className="flex flex-col items-start gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge status={video.status} />
              {statusLabel ? <span className="text-xs font-medium text-muted-foreground">{statusLabel}</span> : null}
            </div>
            {showProgress ? <Progress className="w-full" value={progress ?? 0} /> : null}
            {(canRetry || canReset || canCancel) ? (
              <div className="flex flex-wrap gap-2">
                {canRetry ? (
                  <Button onClick={handleRetry} loading={isRetrying} size="sm">
                    {!isRetrying ? <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" /> : null}
                    Retry
                  </Button>
                ) : null}
                {canReset ? (
                  <Button onClick={handleReset} loading={isResetting} size="sm" variant="outline">
                    {!isResetting ? <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" /> : null}
                    Reset
                  </Button>
                ) : null}
                {canCancel ? (
                  <Button onClick={handleCancel} loading={isCancelling} size="sm" variant="destructive">
                    {!isCancelling ? <XCircle className="h-3.5 w-3.5" aria-hidden="true" /> : null}
                    Cancel
                  </Button>
                ) : null}
              </div>
            ) : null}
          </div>

          <div className="text-sm font-medium text-muted-foreground md:text-card-foreground">
            {video.duration_formatted || (video.status === "completed" ? "Unknown" : "Processing")}
          </div>

          <div className="text-sm font-medium text-muted-foreground md:text-card-foreground">
            {formatFileSize(video.file_size)}
          </div>

          <div className={cn("text-sm font-medium text-muted-foreground", video.status === "processing" && "md:text-info")}>
            {formatDistanceToNow(new Date(video.created_at), { addSuffix: true })}
          </div>
        </div>
      </div>
    </Link>
  );
}
