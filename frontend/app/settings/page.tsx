"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Zap, Settings } from "lucide-react";

export default function SettingsPage() {
  const { data: settings, isLoading: settingsLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const { data: hardware, isLoading: hardwareLoading } = useQuery({
    queryKey: ["hardware"],
    queryFn: api.getHardwareInfo,
  });

  if (settingsLoading || hardwareLoading) {
    return <div className="py-12 text-center text-slate-500 dark:text-slate-400">Loading...</div>;
  }

  const visionLabel = settings?.models.vision_endpoint_url
    ? `${settings.models.vision_endpoint_model || settings.models.vision_model} (endpoint)`
    : settings?.models.vision_model;
  const summarizationLabel = settings?.models.summarization_endpoint_url
    ? `${settings.models.summarization_endpoint_model || settings.models.summarization_model} (endpoint)`
    : settings?.models.summarization_model;

  return (
    <div className="space-y-6">
      {/* Models */}
      <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <h2 className="flex items-center text-lg font-semibold text-slate-900 dark:text-slate-100">
          <Zap className="mr-2 h-5 w-5 text-yellow-600" />
          Models
        </h2>
        <div className="mt-4 space-y-3">
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">ASR</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">{settings?.models.asr_model}</span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">ASR Family</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">
              {settings?.models.asr_family}
            </span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">ASR CPU Override</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">
              {settings?.models.granite_force_cpu ? "Enabled" : "Disabled"}
            </span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">Diarization</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">{settings?.models.diarization_model}</span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">Vision</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">{visionLabel}</span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">Summarization</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">{summarizationLabel}</span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">Audio Events</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">{settings?.models.audio_event_model}</span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">ROCm LLM Runtime</span>
            <span className="font-medium capitalize text-slate-900 dark:text-slate-100">
              {settings?.models.rocm_llm_runtime.replaceAll("_", " ")}
            </span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-slate-600 dark:text-slate-300">ROCm Idle Timeout</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">
              {settings?.models.rocm_llm_idle_timeout_s}s
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-600 dark:text-slate-300">Embeddings</span>
            <span className="font-medium text-slate-900 dark:text-slate-100">{settings?.models.embedding_model}</span>
          </div>
        </div>
      </div>

      {/* Processing */}
      <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <h2 className="flex items-center text-lg font-semibold text-slate-900 dark:text-slate-100">
          <Settings className="mr-2 h-5 w-5 text-slate-600 dark:text-slate-300" />
          Processing
        </h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Max Video Length</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {settings?.processing.max_video_length_hours} hours
            </p>
          </div>
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Keyframe Interval</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {settings?.processing.keyframe_extraction_interval} seconds
            </p>
          </div>
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Concurrent Jobs</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {settings?.processing.batch_concurrent_jobs}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
