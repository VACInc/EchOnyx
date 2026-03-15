"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Settings, Zap } from "lucide-react";

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const { data: settings, isLoading: settingsLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });
  const { data: hardware, isLoading: hardwareLoading } = useQuery({
    queryKey: ["hardware"],
    queryFn: api.getHardwareInfo,
  });
  const { data: availableModels } = useQuery({
    queryKey: ["available-models"],
    queryFn: api.getAvailableModels,
  });

  const [asrModel, setAsrModel] = useState("");
  const [plannerEnabled, setPlannerEnabled] = useState(true);
  const [gpuMemoryFraction, setGpuMemoryFraction] = useState("0.75");
  const [memoryCeilingGb, setMemoryCeilingGb] = useState("");
  const [duplicatePolicy, setDuplicatePolicy] = useState("collapse_exact");
  const [duplicateExactThreshold, setDuplicateExactThreshold] = useState("0.95");
  const [duplicateProbableThreshold, setDuplicateProbableThreshold] = useState("0.85");
  const [saveMessage, setSaveMessage] = useState("");

  useEffect(() => {
    if (!settings) return;
    setAsrModel(settings.models.asr_model);
    setPlannerEnabled(settings.runtime_planner.enabled);
    setGpuMemoryFraction(settings.runtime_planner.gpu_memory_fraction.toString());
    setMemoryCeilingGb(
      settings.runtime_planner.memory_ceiling_gb === null
        ? ""
        : settings.runtime_planner.memory_ceiling_gb.toString()
    );
    setDuplicatePolicy(settings.duplicates.policy);
    setDuplicateExactThreshold(settings.duplicates.exact_threshold.toString());
    setDuplicateProbableThreshold(settings.duplicates.probable_threshold.toString());
  }, [settings]);

  const saveMutation = useMutation({
    mutationFn: () =>
      api.updateSettings({
        asr_model: asrModel,
        runtime_planner_enabled: plannerEnabled,
        gpu_memory_fraction: Number(gpuMemoryFraction),
        runtime_memory_ceiling_gb: memoryCeilingGb === "" ? null : Number(memoryCeilingGb),
        duplicate_detection_policy: duplicatePolicy,
        duplicate_exact_threshold: Number(duplicateExactThreshold),
        duplicate_probable_threshold: Number(duplicateProbableThreshold),
      }),
    onSuccess: async () => {
      setSaveMessage("Saved");
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["settings"] }),
        queryClient.invalidateQueries({ queryKey: ["hardware"] }),
      ]);
    },
    onError: (error) => {
      setSaveMessage(error instanceof Error ? error.message : "Save failed");
    },
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
  const runtimePlan = settings?.runtime_planner;

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <div className="flex items-center justify-between gap-4">
          <h2 className="flex items-center text-lg font-semibold text-slate-900 dark:text-slate-100">
            <Zap className="mr-2 h-5 w-5 text-yellow-600" />
            Models
          </h2>
          <button
            type="button"
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
            disabled={saveMutation.isPending}
            onClick={() => {
              setSaveMessage("");
              saveMutation.mutate();
            }}
          >
            {saveMutation.isPending ? "Saving..." : "Save Settings"}
          </button>
        </div>
        <div className="mt-4 space-y-4">
          <label className="block">
            <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">ASR Model</span>
            <select
              value={asrModel}
              onChange={(event) => setAsrModel(event.target.value)}
              className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
            >
              {availableModels?.asr.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">Runtime Planner</span>
              <select
                value={plannerEnabled ? "enabled" : "disabled"}
                onChange={(event) => setPlannerEnabled(event.target.value === "enabled")}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              >
                <option value="enabled">Enabled</option>
                <option value="disabled">Disabled</option>
              </select>
            </label>
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">GPU Memory Fraction</span>
              <input
                type="number"
                min="0.1"
                max="1"
                step="0.05"
                value={gpuMemoryFraction}
                onChange={(event) => setGpuMemoryFraction(event.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              />
            </label>
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">Memory Ceiling (GB)</span>
              <input
                type="number"
                min="0"
                step="1"
                placeholder="Auto"
                value={memoryCeilingGb}
                onChange={(event) => setMemoryCeilingGb(event.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              />
            </label>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <p className="text-sm text-slate-500 dark:text-slate-400">Current ASR Family</p>
              <p className="mt-1 font-medium text-slate-900 dark:text-slate-100">{settings?.models.asr_family}</p>
            </div>
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">Duplicate Policy</span>
              <select
                value={duplicatePolicy}
                onChange={(event) => setDuplicatePolicy(event.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              >
                <option value="off">Off</option>
                <option value="warn">Warn only</option>
                <option value="collapse_exact">Collapse exact duplicates</option>
                <option value="collapse_probable">Collapse probable duplicates</option>
              </select>
            </label>
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">Exact Threshold</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={duplicateExactThreshold}
                onChange={(event) => setDuplicateExactThreshold(event.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              />
            </label>
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">Probable Threshold</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={duplicateProbableThreshold}
                onChange={(event) => setDuplicateProbableThreshold(event.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              />
            </label>
          </div>
          {saveMessage ? (
            <p className="text-sm text-slate-600 dark:text-slate-300">{saveMessage}</p>
          ) : null}
          <div className="space-y-3">
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
      </div>

      <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <h2 className="flex items-center text-lg font-semibold text-slate-900 dark:text-slate-100">
          <Settings className="mr-2 h-5 w-5 text-slate-600 dark:text-slate-300" />
          Runtime Plan
        </h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Placement</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {runtimePlan?.placement_mode.replaceAll("_", " ")}
            </p>
          </div>
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Budget</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {runtimePlan?.effective_memory_budget_gb} GB / {runtimePlan?.available_accelerator_memory_gb} GB free
            </p>
          </div>
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Worker Loading</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {runtimePlan?.worker_model_loading} ({runtimePlan?.worker_execution_mode})
            </p>
          </div>
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Resident Models</p>
            <p className="mt-2 text-sm text-slate-800 dark:text-slate-200">
              {runtimePlan?.keep_resident_models.length
                ? runtimePlan.keep_resident_models.join(", ")
                : "No resident worker models planned."}
            </p>
            <div className="mt-4 space-y-2 text-sm text-slate-600 dark:text-slate-300">
              <p>Endpoint loading: {runtimePlan?.endpoint_model_loading}</p>
              <p>Endpoint hot residency: {runtimePlan?.can_keep_endpoint_models_loaded ? "allowed" : "not planned"}</p>
              <p>Idle teardown required: {runtimePlan?.requires_endpoint_idle_teardown ? "yes" : "no"}</p>
              <p>Shutdown after request: {runtimePlan?.shutdown_endpoint_after_request ? "yes" : "no"}</p>
              <p>GPU count: {runtimePlan?.accelerator_count ?? hardware?.nvidia_gpus.length ?? 0}</p>
              {runtimePlan?.preferred_worker_devices.length ? (
                <p>Worker placement: {runtimePlan.preferred_worker_devices.join(", ")}</p>
              ) : null}
              {runtimePlan?.preferred_endpoint_devices.length ? (
                <p>Endpoint placement: {runtimePlan.preferred_endpoint_devices.join(", ")}</p>
              ) : null}
              {runtimePlan?.preferred_model_devices && Object.keys(runtimePlan.preferred_model_devices).length ? (
                Object.entries(runtimePlan.preferred_model_devices).map(([key, devices]) => (
                  <p key={key}>
                    {key} placement: {devices.join(", ")}
                  </p>
                ))
              ) : null}
            </div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Planner Notes</p>
            <div className="mt-2 space-y-2 text-sm text-slate-800 dark:text-slate-200">
              {runtimePlan?.notes.length ? (
                runtimePlan.notes.map((note) => <p key={note}>{note}</p>)
              ) : (
                <p>No planner warnings.</p>
              )}
            </div>
          </div>
        </div>
      </div>

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
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Detected Backend</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {hardware?.active_backend}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
