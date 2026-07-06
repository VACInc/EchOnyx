"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, Clock, Info, Upload, Video } from "lucide-react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { VideoCard } from "@/components/video-card";
import { useUploadModal } from "@/components/upload-modal";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { StatusBadge } from "@/components/ui/status-badge";
import { Tooltip } from "@/components/ui/tooltip";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { useToast } from "@/components/ui/toast";

type Job = Awaited<ReturnType<typeof api.getJobs>>["jobs"][number];
type ModelStatus = Awaited<ReturnType<typeof api.getModelStatus>>;
type ModelEntry = NonNullable<ModelStatus["models"][string]>;

const modelStages = [
  {
    key: "whisper",
    label: "Transcription",
    getConfiguredModel: (settings: Awaited<ReturnType<typeof api.getSettings>> | undefined) => settings?.models.asr_model,
  },
  {
    key: "diarization",
    label: "Diarization",
    getConfiguredModel: (settings: Awaited<ReturnType<typeof api.getSettings>> | undefined) => settings?.models.diarization_model,
  },
  {
    key: "vision",
    label: "Vision",
    getConfiguredModel: (settings: Awaited<ReturnType<typeof api.getSettings>> | undefined) =>
      settings?.models.vision_endpoint_model || settings?.models.vision_model,
  },
  {
    key: "summarization",
    label: "Summarization",
    getConfiguredModel: (settings: Awaited<ReturnType<typeof api.getSettings>> | undefined) =>
      settings?.models.summarization_endpoint_model || settings?.models.summarization_model,
  },
  {
    key: "embedding",
    label: "Embeddings",
    getConfiguredModel: (settings: Awaited<ReturnType<typeof api.getSettings>> | undefined) => settings?.models.embedding_model,
  },
];

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function modelProgress(model: ModelEntry | undefined, modelStatus: ModelStatus | undefined): number | null {
  if (!model) return null;
  const directProgress = model.progress_percent;
  if (typeof directProgress === "number") return directProgress;
  const activeDownload = modelStatus?.active_downloads?.find((download) => download.model_name === model.model_name);
  return typeof activeDownload?.progress_percent === "number" ? activeDownload.progress_percent : null;
}

function formatMemory(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" ? `${value.toFixed(digits)} GB` : "Unknown";
}

function RecentVideoSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-card p-4 shadow-sm">
      <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_150px_120px_120px_140px] md:items-center">
        <div className="flex items-start gap-3">
          <Skeleton className="h-10 w-10 shrink-0 rounded-lg" />
          <div className="min-w-0 flex-1 space-y-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-5 w-40 rounded-full" />
          </div>
        </div>
        <Skeleton className="h-6 w-28 rounded-full" />
        <Skeleton className="h-4 w-20" />
        <Skeleton className="h-4 w-16" />
        <Skeleton className="h-4 w-24" />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [hardwareDetailsOpen, setHardwareDetailsOpen] = useState(false);
  const { openModal } = useUploadModal();
  const confirm = useConfirm();
  const toast = useToast();
  const queryClient = useQueryClient();

  const videosQuery = useQuery({
    queryKey: ["videos"],
    queryFn: () => api.getVideos({ page: 1, pageSize: 5 }),
    refetchInterval: 5000,
  });

  const videoStatsQuery = useQuery({
    queryKey: ["videoStats"],
    queryFn: api.getVideoStats,
    refetchInterval: 5000,
  });

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const hardwareQuery = useQuery({
    queryKey: ["hardware"],
    queryFn: api.getHardwareInfo,
  });

  const modelStatusQuery = useQuery({
    queryKey: ["modelStatus"],
    queryFn: api.getModelStatus,
    refetchInterval: 2000,
  });

  const processingJobsQuery = useQuery({
    queryKey: ["jobs", "processing"],
    queryFn: () => api.getJobs({ status: "processing", pageSize: 50 }),
    refetchInterval: 2000,
  });

  const queuedJobsQuery = useQuery({
    queryKey: ["jobs", "queued"],
    queryFn: () => api.getJobs({ status: "queued", pageSize: 50 }),
    refetchInterval: 2000,
  });

  const cancelOrphanedMutation = useMutation({
    mutationFn: api.cancelOrphanedJobs,
    onSuccess: async (data) => {
      await queryClient.invalidateQueries({ queryKey: ["jobs"] });
      const count = data?.cancelled ?? 0;
      toast({
        title: count > 0 ? "Orphaned jobs cancelled" : "No orphaned jobs found",
        description: count > 0 ? `Cancelled ${count} orphaned job(s).` : "Queued and processing jobs are already aligned.",
        variant: count > 0 ? "success" : "info",
      });
    },
    onError: (error) => {
      toast({
        title: "Cleanup failed",
        description: getErrorMessage(error, "Failed to cancel orphaned jobs"),
        variant: "error",
      });
    },
  });

  const workloadJobs = useMemo(() => {
    return [...(processingJobsQuery.data?.jobs ?? []), ...(queuedJobsQuery.data?.jobs ?? [])];
  }, [processingJobsQuery.data?.jobs, queuedJobsQuery.data?.jobs]);

  const jobByVideoId = useMemo(() => {
    const map: Record<string, Job> = {};
    for (const job of workloadJobs) {
      map[job.video_id] = job;
    }
    return map;
  }, [workloadJobs]);

  const stats = [
    {
      label: "Completed",
      value: videoStatsQuery.data?.completed ?? 0,
      icon: Video,
      loading: videoStatsQuery.isLoading,
    },
    {
      label: "Workload",
      value: videoStatsQuery.data?.workload ?? 0,
      icon: Clock,
      loading: videoStatsQuery.isLoading,
    },
  ];

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        {stats.map((stat) => {
          const isWorkload = stat.label === "Workload";
          return (
            <Card key={stat.label} className="p-5">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{stat.label}</p>
                  {stat.loading ? <Skeleton className="mt-3 h-8 w-16" /> : <p className="mt-2 text-2xl font-semibold">{stat.value}</p>}
                  {videoStatsQuery.isError ? (
                    <p className="mt-2 text-xs text-destructive">
                      {getErrorMessage(videoStatsQuery.error, "Stats failed to load.")}
                    </p>
                  ) : null}
                </div>
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
                  <stat.icon className="h-5 w-5" aria-hidden="true" />
                </div>
              </div>
              {isWorkload ? (
                <Button
                  className="mt-4"
                  disabled={cancelOrphanedMutation.isPending}
                  loading={cancelOrphanedMutation.isPending}
                  onClick={async () => {
                    const confirmed = await confirm({
                      title: "Cancel orphaned jobs?",
                      description: "Cancel queued or processing jobs that no longer have a matching video.",
                      confirmLabel: "Clear orphaned",
                      cancelLabel: "Keep jobs",
                    });
                    if (confirmed) {
                      cancelOrphanedMutation.mutate();
                    }
                  }}
                  size="sm"
                  variant="outline"
                >
                  Clear orphaned
                </Button>
              ) : null}
            </Card>
          );
        })}

        <Card className="p-5 lg:col-span-2">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Models</p>
              <h2 className="mt-1 text-lg font-semibold">Runtime model status</h2>
            </div>
            {(modelStatusQuery.data?.active_downloads?.length ?? 0) > 0 ? (
              <Badge variant="info">{modelStatusQuery.data?.active_downloads.length} downloading</Badge>
            ) : null}
          </div>

          {modelStatusQuery.isLoading || settingsQuery.isLoading ? (
            <div className="mt-4 space-y-3">
              {modelStages.map((stage) => (
                <div key={stage.key} className="rounded-lg border border-border p-3">
                  <div className="flex items-center justify-between gap-4">
                    <div className="min-w-0 flex-1 space-y-2">
                      <Skeleton className="h-4 w-32" />
                      <Skeleton className="h-3 w-48" />
                    </div>
                    <Skeleton className="h-6 w-24 rounded-full" />
                  </div>
                </div>
              ))}
            </div>
          ) : modelStatusQuery.isError || settingsQuery.isError ? (
            <ErrorState
              className="mt-4"
              title="Model status failed to load"
              message={getErrorMessage(modelStatusQuery.error ?? settingsQuery.error, "Refresh to try loading model status again.")}
              onRetry={() => {
                modelStatusQuery.refetch();
                settingsQuery.refetch();
              }}
            />
          ) : (
            <div className="mt-4 space-y-3">
              {modelStages.map((stage) => {
                const model = modelStatusQuery.data?.models?.[stage.key];
                const configuredModel = stage.getConfiguredModel(settingsQuery.data) ?? model?.model_name ?? "Not configured";
                const progress = modelProgress(model, modelStatusQuery.data);
                return (
                  <div key={stage.key} className="rounded-lg border border-border p-3">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <h3 className="text-sm font-semibold">{stage.label}</h3>
                          <Tooltip content={model?.error ? `Error: ${model.error}` : `Configured model: ${configuredModel}`}>
                            <span tabIndex={0} className="inline-flex rounded-full text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                              <Info className="h-3.5 w-3.5" aria-hidden="true" />
                            </span>
                          </Tooltip>
                        </div>
                        <p className="mt-1 truncate text-sm text-muted-foreground">{configuredModel}</p>
                      </div>
                      <Tooltip content={model?.error || `${stage.label} is ${model?.status ?? "unknown"}.`}>
                        <StatusBadge status={model?.status ?? "offline"} tabIndex={0} />
                      </Tooltip>
                    </div>
                    {model?.status === "downloading" && progress !== null ? (
                      <div className="mt-3 space-y-1">
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>Download progress</span>
                          <span>{progress.toFixed(0)}%</span>
                        </div>
                        <Progress value={progress} />
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      </div>

      <Card className="p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Hardware</p>
            <h2 className="mt-1 text-lg font-semibold">{hardwareQuery.data?.active_profile ?? settingsQuery.data?.hardware_profile ?? "Detecting profile"}</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-3 lg:w-2/3">
            <div className="rounded-lg border border-border p-3">
              <p className="text-xs text-muted-foreground">GPU backend</p>
              <p className="mt-1 truncate text-sm font-semibold">{hardwareQuery.data?.active_backend ?? settingsQuery.data?.gpu_backend ?? "Detecting"}</p>
            </div>
            <div className="rounded-lg border border-border p-3">
              <p className="text-xs text-muted-foreground">Loading strategy</p>
              <p className="mt-1 truncate text-sm font-semibold">{hardwareQuery.data?.model_loading_strategy ?? settingsQuery.data?.model_loading ?? "Detecting"}</p>
            </div>
            <div className="rounded-lg border border-border p-3">
              <p className="text-xs text-muted-foreground">Memory</p>
              <p className="mt-1 truncate text-sm font-semibold">
                {hardwareQuery.data?.unified_memory_gb
                  ? `${formatMemory(hardwareQuery.data.unified_memory_gb, 0)} unified`
                  : `${formatMemory(hardwareQuery.data?.total_vram_gb)} VRAM`}
              </p>
            </div>
          </div>
        </div>

        {hardwareQuery.isLoading || settingsQuery.isLoading ? (
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <Skeleton className="h-16" />
            <Skeleton className="h-16" />
            <Skeleton className="h-16" />
          </div>
        ) : hardwareQuery.isError || settingsQuery.isError ? (
          <ErrorState
            className="mt-4"
            title="Hardware details failed to load"
            message={getErrorMessage(hardwareQuery.error ?? settingsQuery.error, "Refresh to try loading hardware details again.")}
            onRetry={() => {
              hardwareQuery.refetch();
              settingsQuery.refetch();
            }}
          />
        ) : (
          <details
            className="mt-4 rounded-lg border border-border bg-muted/30"
            open={hardwareDetailsOpen}
            onToggle={(event) => setHardwareDetailsOpen(event.currentTarget.open)}
          >
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              Runtime details
              <ChevronDown className={cn("h-4 w-4 transition", hardwareDetailsOpen && "rotate-180")} aria-hidden="true" />
            </summary>
            <div className="grid gap-3 border-t border-border p-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
              <div>
                <p className="text-xs text-muted-foreground">Whisper backend</p>
                <p className="mt-1 font-medium">{hardwareQuery.data?.whisper_backend ?? "Unknown"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Available VRAM</p>
                <p className="mt-1 font-medium">{formatMemory(hardwareQuery.data?.available_vram_gb)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Runtime planner</p>
                <p className="mt-1 font-medium">{hardwareQuery.data?.runtime_planner_enabled ? "Enabled" : "Disabled"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">ROCm runtime</p>
                <p className="mt-1 font-medium">{hardwareQuery.data?.rocm_llm_runtime ?? "Unknown"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Memory ceiling</p>
                <p className="mt-1 font-medium">{formatMemory(hardwareQuery.data?.runtime_memory_ceiling_gb)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">GPU memory fraction</p>
                <p className="mt-1 font-medium">{hardwareQuery.data?.gpu_memory_fraction ? `${Math.round(hardwareQuery.data.gpu_memory_fraction * 100)}%` : "Unknown"}</p>
              </div>
              {[...(hardwareQuery.data?.nvidia_gpus ?? []), ...(hardwareQuery.data?.amd_gpus ?? [])].length > 0 ? (
                <div className="sm:col-span-2 lg:col-span-3">
                  <p className="text-xs text-muted-foreground">Detected GPUs</p>
                  <ul className="mt-2 grid gap-2">
                    {[...(hardwareQuery.data?.nvidia_gpus ?? []), ...(hardwareQuery.data?.amd_gpus ?? [])].map((gpu, index) => (
                      <li key={`${gpu.name}-${index}`} className="flex items-center justify-between rounded-md bg-card px-3 py-2">
                        <span className="min-w-0 truncate">{gpu.name}</span>
                        <span className="shrink-0 text-muted-foreground">{formatMemory(gpu.vram_gb)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </details>
        )}
      </Card>

      <Card className="p-6">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">Recent videos</h2>
            <p className="mt-1 text-sm text-muted-foreground">Latest uploads and processing activity.</p>
          </div>
          <Link href="/videos" className="text-sm font-medium text-primary hover:underline">
            View all
          </Link>
        </div>

        {videosQuery.isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, index) => (
              <RecentVideoSkeleton key={index} />
            ))}
          </div>
        ) : videosQuery.isError ? (
          <ErrorState
            title="Videos failed to load"
            message={getErrorMessage(videosQuery.error, "Refresh to try loading recent videos again.")}
            onRetry={() => videosQuery.refetch()}
          />
        ) : videosQuery.data?.videos?.length === 0 ? (
          <EmptyState
            icon={<Upload className="h-6 w-6" aria-hidden="true" />}
            headline="No videos yet"
            hint="Upload a video to start transcription, summaries, search, and action extraction."
            action={
              <Button onClick={openModal}>
                <Upload className="h-4 w-4" aria-hidden="true" />
                Upload
              </Button>
            }
          />
        ) : (
          <div className="space-y-3">
            {videosQuery.data?.videos?.map((video) => (
              <VideoCard key={video.id} video={video} job={jobByVideoId[video.id]} />
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
