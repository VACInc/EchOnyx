"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Cpu, HardDrive, Zap, Settings } from "lucide-react";

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
    return <div className="py-12 text-center text-stone-500 dark:text-slate-400">Loading...</div>;
  }

  const visionLabel = settings?.models.vision_endpoint_url
    ? `${settings.models.vision_endpoint_model || settings.models.vision_model} (endpoint)`
    : settings?.models.vision_model;
  const summarizationLabel = settings?.models.summarization_endpoint_url
    ? `${settings.models.summarization_endpoint_model || settings.models.summarization_model} (endpoint)`
    : settings?.models.summarization_model;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold text-stone-900 dark:text-slate-100">Settings</h1>
        <p className="mt-1 text-sm text-stone-500 dark:text-slate-400">
          System configuration and hardware info
        </p>
      </div>

      {/* Hardware Info */}
      <div className="rounded-2xl border border-stone-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <h2 className="flex items-center text-lg font-semibold text-stone-900 dark:text-slate-100">
          <Cpu className="mr-2 h-5 w-5 text-blue-600" />
          Hardware
        </h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-stone-500 dark:text-slate-400">Profile</p>
            <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
              {hardware?.active_profile}
            </p>
          </div>
          <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-stone-500 dark:text-slate-400">GPU Backend</p>
            <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
              {hardware?.active_backend}
            </p>
          </div>
          <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-stone-500 dark:text-slate-400">Whisper Backend</p>
            <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
              {hardware?.whisper_backend}
            </p>
          </div>
          <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-stone-500 dark:text-slate-400">Loading Strategy</p>
            <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
              {hardware?.model_loading_strategy}
            </p>
          </div>
          <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-stone-500 dark:text-slate-400">Total VRAM</p>
            <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
              {hardware?.total_vram_gb?.toFixed(1) ?? 0} GB
            </p>
          </div>
          {hardware?.unified_memory_gb && (
            <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
              <p className="text-sm text-stone-500 dark:text-slate-400">Unified Memory</p>
              <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
                {hardware.unified_memory_gb.toFixed(0)} GB
              </p>
            </div>
          )}
        </div>

        {/* GPU List */}
        {(hardware?.nvidia_gpus?.length || hardware?.amd_gpus?.length) && (
          <div className="mt-4">
            <p className="text-sm font-medium text-stone-700 dark:text-slate-200">Detected GPUs</p>
            <ul className="mt-2 space-y-2">
              {hardware?.nvidia_gpus?.map((gpu, idx) => (
                <li
                  key={`nvidia-${idx}`}
                  className="flex items-center justify-between rounded bg-emerald-50 px-3 py-2 dark:bg-emerald-500/10"
                >
                  <span className="text-sm text-emerald-800 dark:text-emerald-200">{gpu.name}</span>
                  <span className="text-sm text-emerald-600 dark:text-emerald-300">
                    {gpu.vram_gb.toFixed(1)} GB
                  </span>
                </li>
              ))}
              {hardware?.amd_gpus?.map((gpu, idx) => (
                <li
                  key={`amd-${idx}`}
                  className="flex items-center justify-between rounded bg-red-50 px-3 py-2 dark:bg-red-500/10"
                >
                  <span className="text-sm text-red-800 dark:text-red-200">{gpu.name}</span>
                  <span className="text-sm text-red-600 dark:text-red-300">
                    {gpu.vram_gb.toFixed(1)} GB
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Models */}
      <div className="rounded-2xl border border-stone-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <h2 className="flex items-center text-lg font-semibold text-stone-900 dark:text-slate-100">
          <Zap className="mr-2 h-5 w-5 text-yellow-600" />
          Models
        </h2>
        <div className="mt-4 space-y-3">
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-stone-600 dark:text-slate-300">Transcription</span>
            <span className="font-medium text-stone-900 dark:text-slate-100">{settings?.models.whisper_model}</span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-stone-600 dark:text-slate-300">Transcription Fallback</span>
            <span className="font-medium text-stone-900 dark:text-slate-100">
              {settings?.models.transcription_fallback_model}{" "}
              {settings?.models.transcription_fallback_enabled ? "(enabled)" : "(disabled)"}
            </span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-stone-600 dark:text-slate-300">Granite Force CPU</span>
            <span className="font-medium text-stone-900 dark:text-slate-100">
              {settings?.models.granite_force_cpu ? "Enabled" : "Disabled"}
            </span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-stone-600 dark:text-slate-300">Diarization</span>
            <span className="font-medium text-stone-900 dark:text-slate-100">{settings?.models.diarization_model}</span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-stone-600 dark:text-slate-300">Vision</span>
            <span className="font-medium text-stone-900 dark:text-slate-100">{visionLabel}</span>
          </div>
          <div className="flex items-center justify-between border-b pb-2">
            <span className="text-stone-600 dark:text-slate-300">Summarization</span>
            <span className="font-medium text-stone-900 dark:text-slate-100">{summarizationLabel}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-stone-600 dark:text-slate-300">Embeddings</span>
            <span className="font-medium text-stone-900 dark:text-slate-100">{settings?.models.embedding_model}</span>
          </div>
        </div>
      </div>

      {/* Processing */}
      <div className="rounded-2xl border border-stone-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <h2 className="flex items-center text-lg font-semibold text-stone-900 dark:text-slate-100">
          <Settings className="mr-2 h-5 w-5 text-stone-600 dark:text-slate-300" />
          Processing
        </h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-stone-500 dark:text-slate-400">Max Video Length</p>
            <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
              {settings?.processing.max_video_length_hours} hours
            </p>
          </div>
          <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-stone-500 dark:text-slate-400">Keyframe Interval</p>
            <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
              {settings?.processing.keyframe_extraction_interval} seconds
            </p>
          </div>
          <div className="rounded-xl bg-stone-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-stone-500 dark:text-slate-400">Concurrent Jobs</p>
            <p className="text-lg font-medium text-stone-900 dark:text-slate-100">
              {settings?.processing.batch_concurrent_jobs}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
