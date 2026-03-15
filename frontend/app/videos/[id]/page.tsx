"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatTimestamp } from "@/lib/utils";
import {
  ArrowLeft,
  Download,
  Clock,
  Users,
  FileText,
  Image as ImageIcon,
  CheckCircle,
  List,
  Plus,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { use, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

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

const formatDuration = (seconds?: number) => {
  if (seconds === undefined || Number.isNaN(seconds)) {
    return "--";
  }
  return formatTimestamp(seconds);
};

const normalizeActionItemText = (value: string) => value.trim().replace(/\s+/g, " ").toLowerCase();

export default function VideoDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const [activeTab, setActiveTab] = useState<"summary" | "transcript" | "slides">("summary");
  const [now, setNow] = useState(Date.now());
  const [tagInput, setTagInput] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [isSavingTags, setIsSavingTags] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [todoInput, setTodoInput] = useState("");
  const queryClient = useQueryClient();
  const router = useRouter();

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const { data: video, isLoading: isVideoLoading } = useQuery({
    queryKey: ["video", id],
    queryFn: () => api.getVideo(id),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 2000;
      return data.status === "processing" || data.status === "queued" ? 2000 : false;
    },
  });

  useEffect(() => {
    if (video) {
      setTags(video.tags ?? []);
    }
  }, [video]);

  const { data: jobsData } = useQuery({
    queryKey: ["jobs", id],
    queryFn: () => api.getJobs({ videoId: id, pageSize: 1 }),
    enabled: !!id,
    refetchInterval: () => {
      if (!video) return 2000;
      return video.status === "processing" || video.status === "queued" ? 2000 : false;
    },
  });

  const job = useMemo(() => jobsData?.jobs?.[0], [jobsData]);
  const stepProgress = job?.step_progress ?? {};
  const currentStep = job?.current_step ?? null;
  const currentStepData = currentStep ? stepProgress[currentStep] : null;
  const currentStepIndex = currentStepData?.step_index ?? (currentStep ? STEP_ORDER.indexOf(currentStep) + 1 : null);
  const currentStepCount = currentStepData?.step_count ?? STEP_ORDER.length;

  const { data: summary, isLoading: isSummaryLoading } = useQuery({
    queryKey: ["summary", id],
    queryFn: () => api.getSummary(id),
    enabled: video?.status === "completed",
  });

  const { data: settings } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const actionItemsEnabled = settings?.action_items.enabled ?? true;

  const { data: actionItemsData } = useQuery({
    queryKey: ["action-items", { videoId: id }],
    queryFn: () => api.getActionItems({ videoId: id, status: "all", sort: "updated_at", pageSize: 100 }),
    enabled: video?.status === "completed" && actionItemsEnabled,
  });

  const actionItems = useMemo(() => actionItemsData?.items ?? [], [actionItemsData]);
  const existingActionItemTexts = useMemo(
    () => new Set(actionItems.map((item) => normalizeActionItemText(item.text))),
    [actionItems],
  );

  const handleExport = async (format: "md" | "pdf" | "json") => {
    const blob = await api.exportSummary(id, format);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `summary.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDelete = async () => {
    if (!video) return;
    const confirmed = window.confirm(
      "Delete this video and all associated data? This cannot be undone."
    );
    if (!confirmed) return;
    setIsDeleting(true);
    try {
      await api.deleteVideo(video.id);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      router.push("/videos");
    } catch (error) {
      console.error("Failed to delete video:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const saveTags = async (nextTags: string[]) => {
    setIsSavingTags(true);
    try {
      const updated = await api.updateVideoTags(id, nextTags);
      setTags(updated.tags ?? []);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["video", id] });
    } catch (error) {
      console.error("Failed to update tags:", error);
    } finally {
      setIsSavingTags(false);
    }
  };

  const addTagsFromInput = async () => {
    const candidates = tagInput
      .split(",")
      .map((tag) => tag.trim())
      .filter((tag) => tag.length > 0);
    if (candidates.length === 0) {
      return;
    }
    const existingLower = new Set(tags.map((tag) => tag.toLowerCase()));
    const merged = [...tags];
    for (const tag of candidates) {
      if (!existingLower.has(tag.toLowerCase())) {
        merged.push(tag);
        existingLower.add(tag.toLowerCase());
      }
    }
    setTagInput("");
    await saveTags(merged);
  };

  const removeTag = async (tagToRemove: string) => {
    const nextTags = tags.filter((tag) => tag !== tagToRemove);
    await saveTags(nextTags);
  };

  const invalidateActionItems = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["action-items"] }),
      queryClient.invalidateQueries({ queryKey: ["summary", id] }),
    ]);
  };

  const addActionItem = async (text: string, source: "manual" | "summary") => {
    if (!text.trim()) return;
    await api.createActionItem({ videoId: id, text, source });
    setTodoInput("");
    await invalidateActionItems();
  };

  const toggleActionItem = async (actionItemId: string, completed: boolean) => {
    await api.updateActionItem(actionItemId, { completed });
    await invalidateActionItems();
  };

  const deleteActionItem = async (actionItemId: string) => {
    await api.deleteActionItem(actionItemId);
    await invalidateActionItems();
  };

  if (isVideoLoading) {
    return <div className="py-12 text-center text-slate-500 dark:text-slate-400">Loading...</div>;
  }

  if (!video) {
    return <div className="py-12 text-center text-slate-500 dark:text-slate-400">Video not found</div>;
  }

  const headerTitle = summary?.title || video.title || video.original_filename || "Untitled Video";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Link
            href="/videos"
            className="rounded-full p-2 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">
              {headerTitle}
            </h1>
            <div className="mt-1 flex items-center space-x-4 text-sm text-slate-500 dark:text-slate-400">
              <span className="flex items-center">
                <Clock className="mr-1 h-4 w-4" />
                {summary?.duration_formatted || video.duration_formatted || "Processing..."}
              </span>
              <span className="flex items-center">
                <Users className="mr-1 h-4 w-4" />
                {summary ? `${summary.speakers.length} speakers` : "Speakers pending"}
              </span>
            </div>
          </div>
        </div>

        {/* Export */}
        <div className="flex space-x-2">
          <button
            onClick={() => handleExport("md")}
            disabled={!summary}
            className="flex items-center rounded-full border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
          >
            <Download className="mr-2 h-4 w-4" />
            Markdown
          </button>
          <button
            onClick={() => handleExport("pdf")}
            disabled={!summary}
            className="flex items-center rounded-full border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
          >
            <Download className="mr-2 h-4 w-4" />
            PDF
          </button>
          <button
            onClick={() => handleExport("json")}
            disabled={!summary}
            className="flex items-center rounded-full border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
          >
            <Download className="mr-2 h-4 w-4" />
            JSON
          </button>
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="flex items-center rounded-full border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 hover:bg-red-100 disabled:opacity-50 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-300"
          >
            <Trash2 className="mr-2 h-4 w-4" />
            {isDeleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>

      {/* Labels */}
      <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Labels</h2>
            <p className="text-sm text-slate-500 dark:text-slate-400">Add custom labels to organize videos.</p>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {tags.length === 0 && (
            <span className="text-sm text-slate-500 dark:text-slate-400">No labels yet.</span>
          )}
          {tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200"
            >
              {tag}
              <button
                type="button"
                onClick={() => removeTag(tag)}
                disabled={isSavingTags}
                className="ml-2 text-slate-400 hover:text-slate-600 disabled:opacity-50 dark:text-slate-400 dark:hover:text-slate-200"
                aria-label={`Remove ${tag}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
        <div className="mt-4 flex flex-col gap-2 sm:flex-row sm:items-center">
          <input
            type="text"
            value={tagInput}
            onChange={(event) => setTagInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void addTagsFromInput();
              }
            }}
            placeholder="Add label (comma-separated)"
            className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-900/70"
          />
          <button
            type="button"
            onClick={() => void addTagsFromInput()}
            disabled={isSavingTags || tagInput.trim().length === 0}
            className="rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800 disabled:opacity-50 dark:bg-white dark:text-slate-900"
          >
            {isSavingTags ? "Saving..." : "Add label"}
          </button>
        </div>
      </div>

      {video.status !== "completed" && (
        <div className="space-y-4 rounded-2xl border border-blue-200/60 bg-blue-50/70 p-6 shadow-sm dark:border-blue-500/20 dark:bg-blue-500/10">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-blue-900 dark:text-blue-200">Processing status</h2>
              <p className="text-sm text-blue-800 dark:text-blue-200/70">
                {job?.status === "failed"
                  ? "Processing failed"
                  : job?.status === "queued"
                    ? "Queued for processing"
                    : "Processing in progress"}
              </p>
            </div>
            <div className="text-right text-sm text-blue-900 dark:text-blue-200">
              {currentStepIndex ? (
                <div>
                  Step {currentStepIndex} of {currentStepCount}
                </div>
              ) : (
                <div>Step pending</div>
              )}
              {currentStep && (
                <div className="text-xs text-blue-700 dark:text-blue-200/70">
                  {STEP_LABELS[currentStep] || currentStep}
                </div>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm text-blue-800 dark:text-blue-200/70">
              <span>Overall progress</span>
              <span>{job?.progress?.toFixed(0) ?? 0}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-blue-100 dark:bg-blue-900/40">
              <div
                className="h-full rounded-full bg-blue-500 transition-all"
                style={{ width: `${job?.progress ?? 0}%` }}
              />
            </div>
          </div>

          {job?.error_message && (
            <div className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-500/10 dark:text-red-200">
              {job.error_message}
            </div>
          )}

          <div className="rounded-xl bg-white/80 p-4 shadow-sm dark:bg-slate-900/70">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Step details</h3>
            <div className="mt-3 space-y-2 text-sm text-slate-700 dark:text-slate-300">
              {STEP_ORDER.map((stepKey) => {
                const data = stepProgress?.[stepKey] || null;
                const startedAt = data?.started_at ? Date.parse(data.started_at) : null;
                const completedAt = data?.completed_at ? Date.parse(data.completed_at) : null;
                const elapsedSeconds = startedAt && !completedAt
                  ? Math.max(0, Math.floor((now - startedAt) / 1000))
                  : undefined;
                const duration = data?.duration_seconds ?? elapsedSeconds;
                const isActive = currentStep === stepKey;
                const status = data?.completed_at
                  ? "completed"
                  : data?.started_at
                    ? "in progress"
                    : "pending";

                return (
                  <div
                    key={stepKey}
                    className={`flex items-center justify-between rounded-lg px-3 py-2 ${
                      isActive ? "bg-blue-50 dark:bg-blue-500/10" : "bg-slate-50 dark:bg-slate-800/60"
                    }`}
                  >
                    <div>
                      <div className="font-medium text-slate-900 dark:text-slate-100">
                        {STEP_LABELS[stepKey] || stepKey}
                      </div>
                      <div className="text-xs text-slate-500 dark:text-slate-400 capitalize">{status}</div>
                    </div>
                    <div className="text-right text-xs text-slate-600 dark:text-slate-400">
                      <div>{data?.progress !== undefined ? `${data.progress.toFixed(0)}%` : "--"}</div>
                      <div>{formatDuration(duration)}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {video.status === "completed" && (
        <>
          {/* Tabs */}
          <div className="border-b border-slate-200 dark:border-slate-700">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab("summary")}
                className={`flex items-center border-b-2 px-1 py-4 text-sm font-medium ${
                  activeTab === "summary"
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-slate-500 hover:border-slate-300 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                }`}
              >
                <FileText className="mr-2 h-4 w-4" />
                Summary
              </button>
              <button
                onClick={() => setActiveTab("transcript")}
                className={`flex items-center border-b-2 px-1 py-4 text-sm font-medium ${
                  activeTab === "transcript"
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-slate-500 hover:border-slate-300 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                }`}
              >
                <List className="mr-2 h-4 w-4" />
                Transcript ({summary?.transcript.length ?? 0})
              </button>
              <button
                onClick={() => setActiveTab("slides")}
                className={`flex items-center border-b-2 px-1 py-4 text-sm font-medium ${
                  activeTab === "slides"
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-slate-500 hover:border-slate-300 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                }`}
              >
                <ImageIcon className="mr-2 h-4 w-4" />
                Slides ({summary?.slides.length ?? 0})
              </button>
            </nav>
          </div>

          {isSummaryLoading && (
            <div className="py-12 text-center text-slate-500 dark:text-slate-400">Loading summary...</div>
          )}
        </>
      )}

      {/* Content */}
      {video.status === "completed" && activeTab === "summary" && summary?.summary && (
        <div className="space-y-6">
          {/* Executive Summary */}
          <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Executive Summary</h2>
            <p className="mt-2 text-slate-700 dark:text-slate-200">{summary.summary.executive_summary}</p>
          </div>

          {/* Key Points */}
          {summary.summary.key_points.length > 0 && (
            <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Key Points</h2>
              <ul className="mt-2 space-y-2">
                {summary.summary.key_points.map((point, idx) => (
                  <li key={idx} className="flex items-start">
                    <CheckCircle className="mr-2 mt-0.5 h-5 w-5 flex-shrink-0 text-green-500" />
                    <span className="text-slate-700 dark:text-slate-200">{point}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Action Items */}
          {summary.summary.action_items.length > 0 && (
            <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Action Items</h2>
              <ul className="mt-2 space-y-2">
                {summary.summary.action_items.map((item, idx) => (
                  <li key={idx} className="flex items-center justify-between gap-4 rounded-xl bg-slate-50 px-3 py-2 dark:bg-slate-800/60">
                    <span className="text-slate-700 dark:text-slate-200">{item}</span>
                    {actionItemsEnabled ? (
                      <button
                        type="button"
                        disabled={existingActionItemTexts.has(normalizeActionItemText(item))}
                        onClick={() => void addActionItem(item, "summary")}
                        className="rounded-full border border-slate-200 px-3 py-1 text-xs font-medium text-slate-700 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
                      >
                        {existingActionItemTexts.has(normalizeActionItemText(item)) ? "Added" : "Add to todos"}
                      </button>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {actionItemsEnabled && (
            <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Todo List</h2>
                  <p className="text-sm text-slate-500 dark:text-slate-400">Tracked separately so you can filter and check them off later.</p>
                </div>
                <Link href="/todos" className="text-sm font-medium text-blue-600 hover:underline">
                  Open all todos
                </Link>
              </div>
              <div className="mt-4 flex flex-col gap-2 sm:flex-row">
                <input
                  type="text"
                  value={todoInput}
                  onChange={(event) => setTodoInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      void addActionItem(todoInput, "manual");
                    }
                  }}
                  placeholder="Add a manual todo for this video"
                  className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none dark:border-slate-700 dark:bg-slate-900/70"
                />
                <button
                  type="button"
                  onClick={() => void addActionItem(todoInput, "manual")}
                  disabled={todoInput.trim().length === 0}
                  className="inline-flex items-center justify-center rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-white dark:text-slate-900"
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Add todo
                </button>
              </div>
              <div className="mt-4 space-y-2">
                {actionItems.length === 0 ? (
                  <p className="text-sm text-slate-500 dark:text-slate-400">No todos for this video yet.</p>
                ) : (
                  actionItems.map((item) => (
                    <div
                      key={item.id}
                      className="flex items-center justify-between gap-4 rounded-xl bg-slate-50 px-3 py-2 dark:bg-slate-800/60"
                    >
                      <label className="flex flex-1 items-center gap-3">
                        <input
                          type="checkbox"
                          checked={item.completed}
                          onChange={() => void toggleActionItem(item.id, !item.completed)}
                          className="h-4 w-4 rounded"
                        />
                        <span className={item.completed ? "text-slate-400 line-through dark:text-slate-500" : "text-slate-700 dark:text-slate-200"}>
                          {item.text}
                        </span>
                      </label>
                      <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
                        <span className="rounded-full bg-white px-2 py-1 dark:bg-slate-900">{item.source}</span>
                        <button
                          type="button"
                          onClick={() => void deleteActionItem(item.id)}
                          className="text-red-600 hover:underline dark:text-red-300"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Topics */}
          {summary.summary.topics.length > 0 && (
            <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Topic Breakdown</h2>
              <div className="mt-4 space-y-4">
                {summary.summary.topics.map((topic, idx) => (
                  <div key={idx} className="border-l-4 border-blue-500 pl-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-medium text-slate-900 dark:text-slate-100">{topic.topic}</h3>
                      <span className="text-sm text-slate-500 dark:text-slate-400">{topic.timestamp}</span>
                    </div>
                    <p className="mt-1 text-slate-600 dark:text-slate-300">{topic.summary}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {video.status === "completed" && activeTab === "transcript" && summary && (
        <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
          <div className="space-y-4">
            {summary.transcript.map((segment, idx) => (
              <div key={idx} className="flex">
                <span className="w-20 flex-shrink-0 text-sm text-slate-500 dark:text-slate-400">
                  {formatTimestamp(segment.start)}
                </span>
                <div className="flex-1">
                  {segment.speaker && (
                    <span className="font-medium text-blue-600">
                      {segment.speaker}:{" "}
                    </span>
                  )}
                  <span className="text-slate-700 dark:text-slate-200">{segment.text}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {video.status === "completed" && activeTab === "slides" && summary && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {summary.slides.length === 0 ? (
            <p className="col-span-full py-12 text-center text-slate-500 dark:text-slate-400">
              No slides detected
            </p>
          ) : (
            summary.slides.map((slide, idx) => (
              <div key={idx} className="rounded-2xl border border-slate-200/70 bg-white/80 p-4 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
                <div className="mb-2 text-sm text-slate-500 dark:text-slate-400">
                  {formatTimestamp(slide.timestamp)}
                </div>
                {slide.description && (
                  <p className="text-sm text-slate-700 dark:text-slate-200">{slide.description}</p>
                )}
                {slide.ocr_text && (
                  <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">{slide.ocr_text}</p>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
