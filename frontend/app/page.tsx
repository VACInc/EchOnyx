"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { VideoCard } from "@/components/video-card";
import {
  Upload,
  Video,
  Clock,
  Download,
  CheckCircle,
  AlertCircle,
  Loader2,
  Cpu,
  Wifi,
} from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";
import { useUploadModal } from "@/components/upload-modal";

const statusStyles: Record<string, { label: string; className: string }> = {
  loaded: { label: "Loaded", className: "border-emerald-200 bg-emerald-50 text-emerald-700" },
  cached: { label: "Cached", className: "border-emerald-200 bg-emerald-50 text-emerald-700" },
  uncached: { label: "Uncached", className: "border-amber-200 bg-amber-50 text-amber-700" },
  downloading: { label: "Downloading", className: "border-blue-200 bg-blue-50 text-blue-700" },
  failed: { label: "Failed", className: "border-red-200 bg-red-50 text-red-700" },
  online: { label: "Online", className: "border-emerald-200 bg-emerald-50 text-emerald-700" },
  offline: { label: "Offline", className: "border-red-200 bg-red-50 text-red-700" },
};

const modelOrder = ["whisper", "diarization", "vision", "summarization", "embedding"];

const modelLabels: Record<string, string> = {
  whisper: "Transcription",
  diarization: "Diarization",
  vision: "Vision",
  summarization: "Summary",
  embedding: "Embeddings",
};

const modelShortLabels: Record<string, string> = {
  whisper: "T",
  diarization: "D",
  vision: "V",
  summarization: "S",
  embedding: "E",
};

const modelBadge = (modelName: string, modelType: string) => {
  const name = modelName.toLowerCase();
  if (name.includes("qwen")) return { label: "Q", className: "bg-emerald-100 text-emerald-700" };
  if (name.includes("whisper") || name.includes("large-v3")) return { label: "W", className: "bg-blue-100 text-blue-700" };
  if (name.includes("granite")) return { label: "G", className: "bg-slate-200 text-slate-700" };
  if (name.includes("pyannote")) return { label: "P", className: "bg-amber-100 text-amber-700" };
  if (name.includes("nomic")) return { label: "N", className: "bg-indigo-100 text-indigo-700" };
  if (name.includes("gptoss")) return { label: "G", className: "bg-purple-100 text-purple-700" };
  return { label: modelType.slice(0, 1).toUpperCase(), className: "bg-stone-100 text-stone-600" };
};

const statusIcon = (status: string) => {
  switch (status) {
    case "loaded":
    case "cached":
    case "online":
      return CheckCircle;
    case "downloading":
      return Loader2;
    case "uncached":
      return Clock;
    case "offline":
    case "failed":
      return AlertCircle;
    default:
      return Clock;
  }
};

export default function Dashboard() {
  const { openModal } = useUploadModal();

  const { data: videos, isLoading } = useQuery({
    queryKey: ["videos"],
    queryFn: () => api.getVideos({ page: 1, pageSize: 5 }),
    refetchInterval: 5000,
  });

  const { data: settings } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const { data: hardware } = useQuery({
    queryKey: ["hardware"],
    queryFn: api.getHardwareInfo,
  });

  const { data: modelStatus } = useQuery({
    queryKey: ["modelStatus"],
    queryFn: api.getModelStatus,
    refetchInterval: 2000,
  });

  const { data: processingJobs } = useQuery({
    queryKey: ["jobs", "processing"],
    queryFn: () => api.getJobs({ status: "processing", pageSize: 50 }),
    refetchInterval: 2000,
  });

  const downloadCount = modelStatus?.active_downloads?.length ?? 0;
  const modelEntries = modelOrder
    .map((key) => ({ key, data: modelStatus?.models?.[key] }))
    .filter((entry) => entry.data);

  const jobByVideoId = useMemo(() => {
    type ProcessingJob = NonNullable<typeof processingJobs>["jobs"][number];
    const map: Record<string, ProcessingJob> = {};
    if (processingJobs?.jobs) {
      for (const job of processingJobs.jobs) {
        map[job.video_id] = job;
      }
    }
    return map;
  }, [processingJobs]);

  const stats = [
    {
      label: "Total videos",
      value: videos?.total ?? 0,
      icon: Video,
      accent: "text-blue-600",
      bg: "bg-blue-50",
    },
    {
      label: "Active jobs",
      value: processingJobs?.jobs?.length ?? 0,
      icon: Clock,
      accent: "text-amber-600",
      bg: "bg-amber-50",
    },
  ];

  return (
    <div className="space-y-8">
      {/* Quick Stats */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3 lg:grid-cols-4">
        {stats.map((stat, idx) => (
          <div
            key={stat.label}
            className="rounded-2xl border border-stone-200/70 bg-white/80 p-5 shadow-sm animate-fade-up dark:border-slate-700/60 dark:bg-slate-900/70"
            style={{ animationDelay: `${idx * 80}ms` }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400 dark:text-slate-500">
                  {stat.label}
                </p>
                <p className="mt-2 text-2xl font-semibold text-stone-900 dark:text-slate-100">
                  {stat.value}
                </p>
              </div>
              <div className={`rounded-full p-3 ${stat.bg} dark:bg-slate-800`}>
                <stat.icon className={`h-6 w-6 ${stat.accent}`} />
              </div>
            </div>
          </div>
        ))}

        {/* Model Status (compact) */}
        <div className="group relative rounded-2xl border border-stone-200/70 bg-white/80 p-5 shadow-sm animate-fade-up dark:border-slate-700/60 dark:bg-slate-900/70">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400 dark:text-slate-500">
                Model status
              </p>
              {downloadCount > 0 ? (
                <p className="mt-2 text-sm font-semibold text-stone-900 dark:text-slate-100">
                  {`${downloadCount} downloading`}
                </p>
              ) : null}
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {modelEntries.map(({ key, data }) => {
              if (!data) return null;
              const style = statusStyles[data.status] || statusStyles.uncached;
              const shortLabel = modelShortLabels[key] || key.slice(0, 1).toUpperCase();
              const BadgeIcon = statusIcon(data.status);
              return (
                <div
                  key={key}
                  className={`inline-flex flex-col items-center gap-1 rounded-xl border px-2 py-2 text-[10px] font-semibold ${style.className}`}
                >
                  <span className="text-[11px] font-semibold text-stone-700 dark:text-slate-100">
                    {shortLabel}
                  </span>
                  <BadgeIcon className={`h-3.5 w-3.5 ${data.status === "downloading" ? "animate-spin" : ""}`} />
                </div>
              );
            })}
          </div>

          <div className="pointer-events-none absolute left-0 top-full z-20 mt-3 w-72 -translate-y-1 rounded-xl border border-stone-200 bg-white p-4 text-xs text-stone-600 opacity-0 shadow-xl transition duration-200 group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
            <p className="text-xs font-semibold text-stone-700 dark:text-slate-200">Model status</p>
            <div className="mt-3 space-y-2">
              {modelEntries.map(({ key, data }) => {
                if (!data) return null;
                const style = statusStyles[data.status] || statusStyles.uncached;
                const Icon = statusIcon(data.status);
                return (
                  <div key={key} className="flex items-center justify-between">
                    <span className={`inline-flex items-center gap-2 rounded-full border px-2 py-1 text-[10px] font-semibold ${style.className}`}>
                      <Icon className={`h-3 w-3 ${data.status === "downloading" ? "animate-spin" : ""}`} />
                      {modelLabels[key] || key}
                    </span>
                    <span className="text-[10px] uppercase tracking-[0.18em] text-stone-400 dark:text-slate-500">
                      {data.status}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="group relative rounded-2xl border border-stone-200/70 bg-white/80 p-5 shadow-sm animate-fade-up dark:border-slate-700/60 dark:bg-slate-900/70">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400 dark:text-slate-500">
                Hardware
              </p>
              <p className="mt-2 text-lg font-semibold text-stone-900 dark:text-slate-100">
                {settings?.hardware_profile ?? "detecting..."}
              </p>
            </div>
            <div className="rounded-full bg-stone-100 p-3 dark:bg-slate-800">
              <Cpu className="h-6 w-6 text-stone-600 dark:text-slate-300" />
            </div>
          </div>
          <div className="pointer-events-none absolute left-0 top-full z-20 mt-3 w-72 -translate-y-1 rounded-xl border border-stone-200 bg-white p-4 text-sm opacity-0 shadow-xl transition duration-200 group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100 dark:border-slate-700 dark:bg-slate-900">
            <div className="grid gap-2 text-xs text-stone-600 dark:text-slate-300">
              <div className="flex items-center justify-between">
                <span className="text-stone-400 dark:text-slate-500">Profile</span>
                <span>{hardware?.active_profile ?? "detecting"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-stone-400 dark:text-slate-500">GPU backend</span>
                <span>{hardware?.active_backend ?? "detecting"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-stone-400 dark:text-slate-500">Whisper backend</span>
                <span>{hardware?.whisper_backend ?? "detecting"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-stone-400 dark:text-slate-500">Loading strategy</span>
                <span>{hardware?.model_loading_strategy ?? "detecting"}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-stone-400 dark:text-slate-500">Total VRAM</span>
                <span>{hardware?.total_vram_gb?.toFixed(1) ?? 0} GB</span>
              </div>
              {hardware?.unified_memory_gb && (
                <div className="flex items-center justify-between">
                  <span className="text-stone-400 dark:text-slate-500">Unified memory</span>
                  <span>{hardware.unified_memory_gb.toFixed(0)} GB</span>
                </div>
              )}
              {hardware?.nvidia_gpus?.length ? (
                <div className="mt-2">
                  <p className="text-[11px] uppercase tracking-[0.16em] text-stone-400 dark:text-slate-500">
                    GPUs
                  </p>
                  <ul className="mt-2 space-y-1">
                    {hardware.nvidia_gpus.map((gpu, idx) => (
                      <li key={`nvidia-${idx}`} className="flex justify-between">
                        <span>{gpu.name}</span>
                        <span>{gpu.vram_gb.toFixed(1)} GB</span>
                      </li>
                    ))}
                    {hardware.amd_gpus?.map((gpu, idx) => (
                      <li key={`amd-${idx}`} className="flex justify-between">
                        <span>{gpu.name}</span>
                        <span>{gpu.vram_gb.toFixed(1)} GB</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Videos */}
      <div className="rounded-2xl border border-stone-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-stone-900 dark:text-slate-100">
            Recent videos
          </h2>
          <Link
            href="/videos"
            className="text-sm font-medium text-blue-600 hover:text-blue-700"
          >
            View all
          </Link>
        </div>

        {isLoading ? (
          <div className="py-8 text-center text-stone-500 dark:text-slate-400">Loading...</div>
        ) : videos?.videos?.length === 0 ? (
          <div className="py-8 text-center">
            <Video className="mx-auto h-12 w-12 text-stone-400 dark:text-slate-500" />
            <p className="mt-2 text-stone-500 dark:text-slate-400">No videos yet</p>
            <button
              onClick={openModal}
              className="mt-4 inline-flex items-center rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 dark:bg-white dark:text-slate-900"
            >
              <Upload className="mr-2 h-4 w-4" />
              Upload your first video
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {videos?.videos?.map((video) => (
              <VideoCard key={video.id} video={video} job={jobByVideoId[video.id]} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
