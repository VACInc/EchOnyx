"use client";

import { use, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  CheckCircle2,
  Clock,
  Download,
  FileText,
  Image as ImageIcon,
  List,
  Plus,
  RefreshCw,
  Tags,
  Trash2,
  Users,
  XCircle,
} from "lucide-react";

import { api } from "@/lib/api";
import { cn, formatTimestamp } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { StatusBadge } from "@/components/ui/status-badge";
import { Tabs } from "@/components/ui/tabs";
import { TagInput } from "@/components/ui/tag-input";
import { Tooltip } from "@/components/ui/tooltip";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { useToast } from "@/components/ui/toast";

type ExportFormat = "md" | "pdf" | "json";
type Summary = Awaited<ReturnType<typeof api.getSummary>>;
type Slide = Summary["slides"][number];
type SimilarResult = Awaited<ReturnType<typeof api.getSimilarVideos>>["results"][number];

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

const STEP_LABELS: Record<string, string> = {
  audio_extraction: "Extract audio",
  transcription: "Transcription",
  diarization: "Diarization",
  transcript_merge: "Merge transcript",
  frame_extraction: "Extract frames",
  vision_analysis: "Vision analysis",
  summarization: "Summarization",
  embedding: "Embedding",
};

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function formatDuration(seconds?: number) {
  if (seconds === undefined || Number.isNaN(seconds)) {
    return "--";
  }
  return formatTimestamp(seconds);
}

function formatScore(score: number | null | undefined) {
  if (score === null || score === undefined || Number.isNaN(score)) {
    return "Score unavailable";
  }
  return score <= 1 ? `${(score * 100).toFixed(1)}%` : score.toFixed(2);
}

function duplicateLabel(classification: string) {
  return classification
    .split("_")
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}

function normalizeActionItemText(value: string) {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

function VideoDetailSkeleton() {
  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex min-w-0 items-start gap-4">
          <Skeleton className="h-10 w-10 shrink-0 rounded-full" />
          <div className="min-w-0 flex-1 space-y-3">
            <Skeleton className="h-7 w-72 max-w-full" />
            <div className="flex flex-wrap gap-3">
              <Skeleton className="h-5 w-28" />
              <Skeleton className="h-5 w-24" />
            </div>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: 4 }).map((_, index) => (
            <Skeleton key={index} className="h-10 w-28 rounded-lg" />
          ))}
        </div>
      </div>
      <Card className="space-y-4 p-6">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-11 w-full rounded-lg" />
      </Card>
      <Card className="space-y-4 p-6">
        <Skeleton className="h-5 w-44" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-24 w-full" />
      </Card>
    </div>
  );
}

function SummarySkeleton() {
  return (
    <div className="space-y-6">
      {Array.from({ length: 3 }).map((_, index) => (
        <Card key={index} className="space-y-3 p-6">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-2/3" />
        </Card>
      ))}
    </div>
  );
}

function DuplicateBanner({
  duplicateInfo,
}: {
  duplicateInfo: NonNullable<Awaited<ReturnType<typeof api.getVideo>>["duplicate_info"]>;
}) {
  if (!duplicateInfo.classification) {
    return null;
  }

  const matchTitle = duplicateInfo.duplicate_of?.title || "another video";
  const label = duplicateLabel(duplicateInfo.classification);
  const score = formatScore(duplicateInfo.score);
  const duplicateText = `${label} of ${matchTitle}`;

  return (
    <Card className="flex flex-col gap-3 border-warning/40 bg-warning/10 p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <Tooltip content="Duplicate detection suppresses redundant matches so the canonical video stays easier to find.">
            <Badge variant="warning" tabIndex={0}>
              Duplicate
            </Badge>
          </Tooltip>
          <span className="text-sm font-medium text-card-foreground">{duplicateText}</span>
          <Badge variant="outline">{score}</Badge>
          {duplicateInfo.suppressed ? <Badge variant="muted">Suppressed</Badge> : null}
        </div>
        <p className="text-sm text-muted-foreground">
          This video has a duplicate classification. Review the linked source before reprocessing or deleting.
        </p>
      </div>
      {duplicateInfo.duplicate_of?.id ? (
        <Link
          href={`/videos/${duplicateInfo.duplicate_of.id}`}
          className="inline-flex h-9 shrink-0 items-center justify-center rounded-md border border-border bg-card px-3 text-sm font-medium text-card-foreground transition hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
        >
          Open duplicate
        </Link>
      ) : null}
    </Card>
  );
}

function SlideCard({ slide, videoId }: { slide: Slide; videoId: string }) {
  const [imageFailed, setImageFailed] = useState(false);
  const timestamp = formatTimestamp(slide.timestamp);
  const altText = slide.description || `Slide at ${timestamp}`;

  return (
    <Card className="overflow-hidden">
      <div className="aspect-video bg-muted">
        {imageFailed ? (
          <div className="flex h-full items-center justify-center px-4 text-center text-sm text-muted-foreground">
            Slide image unavailable
          </div>
        ) : (
          // eslint-disable-next-line @next/next/no-img-element -- slide images come from the authenticated API origin; next/image optimization would drop session cookies
          <img
            src={api.slideImageUrl(videoId, slide.image_path)}
            alt={altText}
            loading="lazy"
            onError={() => setImageFailed(true)}
            className="h-full w-full object-contain"
          />
        )}
      </div>
      <div className="space-y-3 p-4">
        <Badge variant="outline">{timestamp}</Badge>
        {slide.description ? <p className="text-sm text-card-foreground">{slide.description}</p> : null}
        {slide.ocr_text ? (
          <p className="rounded-md bg-muted p-3 text-xs text-muted-foreground">{slide.ocr_text}</p>
        ) : null}
      </div>
    </Card>
  );
}

function SimilarVideosSection({ videoId }: { videoId: string }) {
  const similarQuery = useQuery({
    queryKey: ["similar-videos", videoId],
    queryFn: () => api.getSimilarVideos(videoId),
  });

  const results = useMemo(
    () => (similarQuery.data?.results ?? []).filter((result) => result.video_id !== videoId),
    [similarQuery.data?.results, videoId],
  );

  return (
    <Card className="space-y-4 p-6">
      <div>
        <h2 className="text-lg font-semibold text-card-foreground">Similar Videos</h2>
        <p className="text-sm text-muted-foreground">Related videos based on searchable transcript and summary context.</p>
      </div>

      {similarQuery.isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, index) => (
            <div key={index} className="space-y-2 rounded-lg border border-border p-3">
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-3 w-full" />
            </div>
          ))}
        </div>
      ) : similarQuery.isError ? (
        <ErrorState
          title="Similar videos unavailable"
          message={getErrorMessage(similarQuery.error, "Try loading related videos again.")}
          retryLabel="Retry"
          onRetry={() => void similarQuery.refetch()}
        />
      ) : results.length === 0 ? (
        <EmptyState
          icon={<List className="h-6 w-6" aria-hidden="true" />}
          headline="No similar videos found"
          hint="This video does not have close searchable matches yet."
        />
      ) : (
        <div className="space-y-3">
          {results.map((result) => (
            <SimilarVideoRow key={`${result.video_id}-${result.timestamp ?? "video"}`} result={result} />
          ))}
        </div>
      )}
    </Card>
  );
}

function SimilarVideoRow({ result }: { result: SimilarResult }) {
  return (
    <Link
      href={`/videos/${result.video_id}`}
      className="block rounded-lg border border-border p-3 transition hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-card-foreground">{result.video_title || "Untitled video"}</h3>
          <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
            {result.context || result.text || "Matched by semantic similarity."}
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          {result.timestamp_formatted ? <Badge variant="outline">{result.timestamp_formatted}</Badge> : null}
          <Badge variant="muted">{formatScore(result.relevance_score)}</Badge>
        </div>
      </div>
    </Link>
  );
}

export default function VideoDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [activeTab, setActiveTab] = useState<"summary" | "transcript" | "slides">("summary");
  const [now, setNow] = useState(Date.now());
  const [tags, setTags] = useState<string[]>([]);
  const [isSavingTags, setIsSavingTags] = useState(false);
  const [exportingFormat, setExportingFormat] = useState<ExportFormat | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [todoInput, setTodoInput] = useState("");
  const queryClient = useQueryClient();
  const router = useRouter();
  const confirm = useConfirm();
  const toast = useToast();

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const videoQuery = useQuery({
    queryKey: ["video", id],
    queryFn: () => api.getVideo(id),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      return data.status === "processing" || data.status === "queued" ? 2000 : false;
    },
  });

  const video = videoQuery.data;

  useEffect(() => {
    if (video) {
      setTags(video.tags ?? []);
    }
  }, [video]);

  const labelsQuery = useQuery({
    queryKey: ["video-labels"],
    queryFn: api.getVideoLabels,
    enabled: !!video,
  });

  const jobsQuery = useQuery({
    queryKey: ["jobs", id],
    queryFn: () => api.getJobs({ videoId: id, pageSize: 1 }),
    enabled: !!id,
    refetchInterval: () => {
      if (!video) return 2000;
      return video.status === "processing" || video.status === "queued" ? 2000 : false;
    },
  });

  const job = useMemo(() => jobsQuery.data?.jobs?.[0], [jobsQuery.data]);
  const stepProgress = job?.step_progress ?? {};
  const currentStep = job?.current_step ?? null;
  const currentStepData = currentStep ? stepProgress[currentStep] : null;
  const currentStepIndex = currentStepData?.step_index ?? (currentStep ? STEP_ORDER.indexOf(currentStep) + 1 : null);
  const currentStepCount = currentStepData?.step_count ?? STEP_ORDER.length;

  const summaryQuery = useQuery({
    queryKey: ["summary", id],
    queryFn: () => api.getSummary(id),
    enabled: video?.status === "completed",
  });

  const summary = summaryQuery.data;

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const actionItemsEnabled = settingsQuery.data?.action_items.enabled ?? true;

  const actionItemsQuery = useQuery({
    queryKey: ["action-items", { videoId: id }],
    queryFn: () => api.getActionItems({ videoId: id, status: "all", sort: "updated_at", pageSize: 100 }),
    enabled: video?.status === "completed" && actionItemsEnabled,
  });

  const actionItems = useMemo(() => actionItemsQuery.data?.items ?? [], [actionItemsQuery.data]);
  const existingActionItemTexts = useMemo(
    () => new Set(actionItems.map((item) => normalizeActionItemText(item.text))),
    [actionItems],
  );

  const labelSuggestions = useMemo(() => {
    const activeLabels = new Set(tags.map((tag) => tag.toLowerCase()));
    return (labelsQuery.data?.labels ?? [])
      .map((label) => label.name)
      .filter((label) => !activeLabels.has(label.toLowerCase()));
  }, [labelsQuery.data?.labels, tags]);

  const invalidateVideoWork = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["videos"] }),
      queryClient.invalidateQueries({ queryKey: ["video", id] }),
      queryClient.invalidateQueries({ queryKey: ["jobs"] }),
      queryClient.invalidateQueries({ queryKey: ["summary", id] }),
    ]);
  };

  const handleExport = async (format: ExportFormat) => {
    if (!summary) return;
    setExportingFormat(format);
    try {
      const blob = await api.exportSummary(id, format);
      if (blob.type.includes("application/json")) {
        const payload = await blob.text().then((value) => JSON.parse(value)).catch(() => null);
        if (payload?.detail && !payload.video_id && !payload.summary) {
          throw new Error(payload.detail);
        }
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `summary.${format}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      toast({
        title: "Export failed",
        description: getErrorMessage(error, "Try exporting the summary again."),
        variant: "error",
      });
    } finally {
      setExportingFormat(null);
    }
  };

  const handleDelete = async () => {
    if (!video) return;
    const confirmed = await confirm({
      title: "Delete this video?",
      description: "Delete this video and all associated data. This cannot be undone.",
      confirmLabel: "Delete",
      cancelLabel: "Keep video",
      destructive: true,
    });
    if (!confirmed) return;
    setIsDeleting(true);
    try {
      await api.deleteVideo(video.id);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      router.push("/videos");
    } catch (error) {
      toast({
        title: "Delete failed",
        description: getErrorMessage(error, "Failed to delete video"),
        variant: "error",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  const saveTags = async (nextTags: string[]) => {
    const previousTags = tags;
    setTags(nextTags);
    setIsSavingTags(true);
    try {
      const updated = await api.updateVideoTags(id, nextTags);
      setTags(updated.tags ?? []);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["videos"] }),
        queryClient.invalidateQueries({ queryKey: ["video", id] }),
        queryClient.invalidateQueries({ queryKey: ["video-labels"] }),
      ]);
    } catch (error) {
      setTags(previousTags);
      toast({
        title: "Labels not saved",
        description: getErrorMessage(error, "Try saving the labels again."),
        variant: "error",
      });
    } finally {
      setIsSavingTags(false);
    }
  };

  const invalidateActionItems = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["action-items"] }),
      queryClient.invalidateQueries({ queryKey: ["summary", id] }),
    ]);
  };

  const addActionItem = async (text: string, source: "manual" | "summary") => {
    if (!text.trim()) return;
    try {
      await api.createActionItem({ videoId: id, text, source });
      setTodoInput("");
      await invalidateActionItems();
    } catch (error) {
      toast({
        title: "Todo not added",
        description: getErrorMessage(error, "Try adding the todo again."),
        variant: "error",
      });
    }
  };

  const toggleActionItem = async (actionItemId: string, completed: boolean) => {
    try {
      await api.updateActionItem(actionItemId, { completed });
      await invalidateActionItems();
    } catch (error) {
      toast({
        title: "Todo not updated",
        description: getErrorMessage(error, "Try updating the todo again."),
        variant: "error",
      });
    }
  };

  const deleteActionItem = async (actionItemId: string) => {
    try {
      await api.deleteActionItem(actionItemId);
      await invalidateActionItems();
    } catch (error) {
      toast({
        title: "Todo not removed",
        description: getErrorMessage(error, "Try removing the todo again."),
        variant: "error",
      });
    }
  };

  const handleRetry = async () => {
    setIsRetrying(true);
    try {
      await api.retryVideo(id);
      await invalidateVideoWork();
      toast({
        title: "Retry started",
        description: "Processing has been queued again.",
        variant: "success",
      });
    } catch (error) {
      toast({
        title: "Retry failed",
        description: getErrorMessage(error, "Try retrying the video again."),
        variant: "error",
      });
    } finally {
      setIsRetrying(false);
    }
  };

  const handleReset = async () => {
    setIsResetting(true);
    try {
      await api.resetVideo(id);
      await invalidateVideoWork();
      toast({
        title: "Video reset",
        description: "Processing state was reset.",
        variant: "success",
      });
    } catch (error) {
      toast({
        title: "Reset failed",
        description: getErrorMessage(error, "Try resetting the video again."),
        variant: "error",
      });
    } finally {
      setIsResetting(false);
    }
  };

  const handleCancel = async () => {
    if (!job?.id) return;
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
      await invalidateVideoWork();
      toast({
        title: "Job cancelled",
        description: "The video processing job was cancelled.",
        variant: "success",
      });
    } catch (error) {
      toast({
        title: "Cancel failed",
        description: getErrorMessage(error, "Try cancelling the job again."),
        variant: "error",
      });
    } finally {
      setIsCancelling(false);
    }
  };

  if (videoQuery.isLoading) {
    return <VideoDetailSkeleton />;
  }

  if (!video) {
    return (
      <EmptyState
        icon={<FileText className="h-6 w-6" aria-hidden="true" />}
        headline="Video not found"
        hint="The video may have been deleted or the link may be out of date."
        action={
          <Link
            href="/videos"
            className="inline-flex h-10 items-center justify-center rounded-lg border border-border bg-card px-4 text-sm font-medium text-card-foreground transition hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            Back to videos
          </Link>
        }
      />
    );
  }

  const headerTitle = summary?.title || video.title || video.original_filename || "Untitled Video";
  const processingStatusText = job?.status === "failed"
    ? "Processing failed"
    : job?.status === "queued" || video.status === "queued"
      ? "Queued for processing"
      : "Processing in progress";
  const canCancel = (video.status === "queued" || video.status === "processing") && Boolean(job?.id);
  const isFailed = video.status === "failed" || job?.status === "failed";

  const summaryContent = summaryQuery.isLoading ? (
    <SummarySkeleton />
  ) : summaryQuery.isError ? (
    <ErrorState
      title="Summary unavailable"
      message={getErrorMessage(summaryQuery.error, "Try loading the summary again.")}
      retryLabel="Retry"
      onRetry={() => void summaryQuery.refetch()}
    />
  ) : summary?.summary ? (
    <div className="space-y-6">
      <Card className="space-y-2 p-6">
        <h2 className="text-lg font-semibold text-card-foreground">Executive Summary</h2>
        <p className="text-sm leading-6 text-card-foreground">{summary.summary.executive_summary}</p>
      </Card>

      {summary.summary.key_points.length > 0 ? (
        <Card className="space-y-3 p-6">
          <h2 className="text-lg font-semibold text-card-foreground">Key Points</h2>
          <ul className="space-y-2">
            {summary.summary.key_points.map((point) => (
              <li key={point} className="flex items-start gap-2 text-sm text-card-foreground">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" aria-hidden="true" />
                <span>{point}</span>
              </li>
            ))}
          </ul>
        </Card>
      ) : null}

      {summary.summary.action_items.length > 0 ? (
        <Card className="space-y-3 p-6">
          <h2 className="text-lg font-semibold text-card-foreground">Action Items</h2>
          <ul className="space-y-2">
            {summary.summary.action_items.map((item) => {
              const alreadyAdded = existingActionItemTexts.has(normalizeActionItemText(item));
              return (
                <li key={item} className="flex flex-col gap-3 rounded-lg bg-muted p-3 sm:flex-row sm:items-center sm:justify-between">
                  <span className="text-sm text-card-foreground">{item}</span>
                  {actionItemsEnabled ? (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={alreadyAdded}
                      onClick={() => void addActionItem(item, "summary")}
                    >
                      {alreadyAdded ? "Added" : "Add to todos"}
                    </Button>
                  ) : null}
                </li>
              );
            })}
          </ul>
        </Card>
      ) : null}

      {actionItemsEnabled ? (
        <Card className="space-y-4 p-6">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-card-foreground">Todo List</h2>
              <p className="text-sm text-muted-foreground">Tracked separately so you can filter and check them off later.</p>
            </div>
            <Link href="/todos" className="text-sm font-medium text-primary hover:underline">
              Open all todos
            </Link>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Field label="Manual todo" className="flex-1">
              <Input
                value={todoInput}
                onChange={(event) => setTodoInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void addActionItem(todoInput, "manual");
                  }
                }}
                placeholder="Add a manual todo for this video"
              />
            </Field>
            <Button
              type="button"
              className="mt-auto"
              onClick={() => void addActionItem(todoInput, "manual")}
              disabled={todoInput.trim().length === 0}
            >
              <Plus className="h-4 w-4" aria-hidden="true" />
              Add todo
            </Button>
          </div>
          <div className="space-y-2">
            {actionItems.length === 0 ? (
              <EmptyState
                icon={<List className="h-6 w-6" aria-hidden="true" />}
                headline="No todos yet"
                hint="Add one manually or save a suggestion from the summary."
              />
            ) : (
              actionItems.map((item) => (
                <div
                  key={item.id}
                  className="flex flex-col gap-3 rounded-lg bg-muted p-3 sm:flex-row sm:items-center sm:justify-between"
                >
                  <label className="flex flex-1 items-center gap-3">
                    <input
                      type="checkbox"
                      checked={item.completed}
                      onChange={() => void toggleActionItem(item.id, !item.completed)}
                      className="h-4 w-4 rounded border-input accent-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                    />
                    <span className={cn("text-sm text-card-foreground", item.completed && "text-muted-foreground line-through")}>
                      {item.text}
                    </span>
                  </label>
                  <div className="flex items-center gap-2">
                    <Badge variant="muted">{item.source}</Badge>
                    <Button type="button" variant="ghost" size="sm" onClick={() => void deleteActionItem(item.id)}>
                      Remove
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      ) : null}

      {summary.summary.topics.length > 0 ? (
        <Card className="space-y-4 p-6">
          <h2 className="text-lg font-semibold text-card-foreground">Topic Breakdown</h2>
          <div className="space-y-4">
            {summary.summary.topics.map((topic) => (
              <div key={`${topic.timestamp}-${topic.topic}`} className="border-l-4 border-primary pl-4">
                <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                  <h3 className="font-medium text-card-foreground">{topic.topic}</h3>
                  <span className="text-sm text-muted-foreground">{topic.timestamp}</span>
                </div>
                <p className="mt-1 text-sm leading-6 text-muted-foreground">{topic.summary}</p>
              </div>
            ))}
          </div>
        </Card>
      ) : null}

    </div>
  ) : (
    <EmptyState
      icon={<FileText className="h-6 w-6" aria-hidden="true" />}
      headline="No summary available"
      hint="The transcript and slides may still be available in the other tabs."
    />
  );

  const transcriptContent = summaryQuery.isLoading ? (
    <SummarySkeleton />
  ) : summary?.transcript.length ? (
    <Card className="p-6">
      <div className="space-y-4">
        {summary.transcript.map((segment, index) => (
          <div key={`${segment.start}-${index}`} className="flex gap-4">
            <span className="w-20 shrink-0 text-sm text-muted-foreground">{formatTimestamp(segment.start)}</span>
            <div className="min-w-0 flex-1 text-sm leading-6 text-card-foreground">
              {segment.speaker ? <span className="font-medium text-primary">{segment.speaker}: </span> : null}
              <span>{segment.text}</span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  ) : (
    <EmptyState
      icon={<List className="h-6 w-6" aria-hidden="true" />}
      headline="No transcript available"
      hint="This video does not have transcript segments to display."
    />
  );

  const slidesContent = summaryQuery.isLoading ? (
    <SummarySkeleton />
  ) : summary?.slides.length ? (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {summary.slides.map((slide, index) => (
        <SlideCard key={`${slide.image_path}-${slide.timestamp}-${index}`} slide={slide} videoId={video.id} />
      ))}
    </div>
  ) : (
    <EmptyState
      icon={<ImageIcon className="h-6 w-6" aria-hidden="true" />}
      headline="No slides detected"
      hint="Slides will appear here when the vision step extracts them."
    />
  );

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex min-w-0 items-start gap-4">
          <Link
            href="/videos"
            aria-label="Back to videos"
            className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-foreground transition hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            <ArrowLeft className="h-5 w-5" aria-hidden="true" />
          </Link>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="min-w-0 text-2xl font-semibold text-foreground">{headerTitle}</h1>
              <StatusBadge status={video.status} />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-1.5">
                <Clock className="h-4 w-4" aria-hidden="true" />
                {summary?.duration_formatted || video.duration_formatted || "Processing"}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Users className="h-4 w-4" aria-hidden="true" />
                {summary ? `${summary.speakers.length} speakers` : "Speakers pending"}
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {(["md", "pdf", "json"] as const).map((format) => (
            <Button
              key={format}
              type="button"
              variant="outline"
              loading={exportingFormat === format}
              disabled={!summary || exportingFormat !== null}
              onClick={() => void handleExport(format)}
            >
              <Download className="h-4 w-4" aria-hidden="true" />
              {format === "md" ? "Markdown" : format.toUpperCase()}
            </Button>
          ))}
          <Button type="button" variant="destructive" loading={isDeleting} onClick={() => void handleDelete()}>
            <Trash2 className="h-4 w-4" aria-hidden="true" />
            Delete
          </Button>
        </div>
      </header>

      {video.duplicate_info?.classification ? <DuplicateBanner duplicateInfo={video.duplicate_info} /> : null}

      <Card className="space-y-4 p-6">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold text-card-foreground">
            <Tags className="h-5 w-5" aria-hidden="true" />
            Labels
          </h2>
          <p className="text-sm text-muted-foreground">Add custom labels to organize videos.</p>
        </div>
        <TagInput
          value={tags}
          suggestions={labelSuggestions}
          placeholder={labelsQuery.isLoading ? "Loading labels" : "Add label"}
          onChange={(nextTags) => void saveTags(nextTags)}
        />
        <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
          <span>{tags.length === 0 ? "No labels yet." : `${tags.length} label${tags.length === 1 ? "" : "s"} applied.`}</span>
          {isSavingTags ? <span>Saving labels...</span> : null}
        </div>
      </Card>

      {video.status !== "completed" ? (
        <Card className="space-y-5 p-6">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-lg font-semibold text-card-foreground">Processing Status</h2>
                <StatusBadge status={job?.status ?? video.status} />
              </div>
              <p className="text-sm text-muted-foreground">{processingStatusText}</p>
            </div>
            <div className="text-sm text-muted-foreground sm:text-right">
              {currentStepIndex ? <div>Step {currentStepIndex} of {currentStepCount}</div> : <div>Step pending</div>}
              {currentStep ? <div className="text-xs">{STEP_LABELS[currentStep] || currentStep}</div> : null}
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <span>Overall progress</span>
              <span>{job?.progress?.toFixed(0) ?? 0}%</span>
            </div>
            <Progress value={job?.progress ?? 0} />
          </div>

          {isFailed ? (
            <div className="space-y-3">
              <ErrorState
                title="Processing failed"
                message={job?.error_message || "Retry processing, or reset the video state before running it again."}
                retryLabel="Retry"
                onRetry={() => void handleRetry()}
              />
              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="outline" loading={isRetrying} onClick={() => void handleRetry()}>
                  <RefreshCw className="h-4 w-4" aria-hidden="true" />
                  Retry
                </Button>
                <Button type="button" variant="destructive" loading={isResetting} onClick={() => void handleReset()}>
                  <XCircle className="h-4 w-4" aria-hidden="true" />
                  Reset
                </Button>
              </div>
            </div>
          ) : null}

          {canCancel ? (
            <Button type="button" variant="outline" loading={isCancelling} onClick={() => void handleCancel()}>
              <XCircle className="h-4 w-4" aria-hidden="true" />
              Cancel job
            </Button>
          ) : null}

          <Card className="space-y-3 bg-muted/40 p-4 shadow-none">
            <h3 className="text-sm font-semibold text-card-foreground">Step Details</h3>
            <div className="space-y-2">
              {STEP_ORDER.map((stepKey) => {
                const data = stepProgress?.[stepKey] || null;
                const startedAt = data?.started_at ? Date.parse(data.started_at) : null;
                const completedAt = data?.completed_at ? Date.parse(data.completed_at) : null;
                const elapsedSeconds = startedAt && !completedAt
                  ? Math.max(0, Math.floor((now - startedAt) / 1000))
                  : undefined;
                const duration = data?.duration_seconds ?? elapsedSeconds;
                const isActive = currentStep === stepKey;
                const stepFailed = isFailed && job?.error_step === stepKey;
                const stepStatus = stepFailed
                  ? "failed"
                  : data?.completed_at
                    ? "completed"
                    : data?.started_at || isActive
                      ? "processing"
                      : "queued";
                const progress = data?.progress ?? (stepStatus === "completed" ? 100 : 0);

                return (
                  <div
                    key={stepKey}
                    className={cn(
                      "grid gap-3 rounded-lg border border-border bg-card p-3 sm:grid-cols-[minmax(0,1fr)_180px_96px] sm:items-center",
                      isActive && "ring-1 ring-ring",
                    )}
                  >
                    <div className="min-w-0">
                      <div className="font-medium text-card-foreground">{STEP_LABELS[stepKey] || stepKey}</div>
                      <div className="mt-1">
                        <StatusBadge status={stepStatus} />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Progress</span>
                        <span>{progress.toFixed(0)}%</span>
                      </div>
                      <Progress value={progress} />
                    </div>
                    <div className="text-sm text-muted-foreground sm:text-right">{formatDuration(duration)}</div>
                  </div>
                );
              })}
            </div>
          </Card>
        </Card>
      ) : (
        <>
          <Tabs
            value={activeTab}
            onValueChange={(value) => setActiveTab(value as typeof activeTab)}
            tabs={[
              {
                id: "summary",
                label: (
                  <span className="inline-flex items-center gap-2">
                    <FileText className="h-4 w-4" aria-hidden="true" />
                    Summary
                  </span>
                ),
                content: summaryContent,
              },
              {
                id: "transcript",
                label: (
                  <span className="inline-flex items-center gap-2">
                    <List className="h-4 w-4" aria-hidden="true" />
                    Transcript ({summary?.transcript.length ?? 0})
                  </span>
                ),
                content: transcriptContent,
              },
              {
                id: "slides",
                label: (
                  <span className="inline-flex items-center gap-2">
                    <ImageIcon className="h-4 w-4" aria-hidden="true" />
                    Slides ({summary?.slides.length ?? 0})
                  </span>
                ),
                content: slidesContent,
              },
            ]}
          />
          <SimilarVideosSection videoId={video.id} />
        </>
      )}
    </div>
  );
}
