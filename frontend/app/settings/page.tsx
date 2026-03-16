"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/components/auth-gate";
import { api } from "@/lib/api";
import { CheckCircle2, Plus, Settings, ShieldCheck, Zap } from "lucide-react";

const MODEL_KEYS = [
  "asr",
  "diarization",
  "vision",
  "summarization",
  "embedding",
  "audio_event",
] as const;

type ModelKey = (typeof MODEL_KEYS)[number];

type ModelOption = {
  name: string;
  size_gb: number;
  recommended: boolean;
};

type ModelSelections = Record<ModelKey, string>;
type CustomModelState = Record<ModelKey, ModelOption[]>;

const MODEL_LABELS: Record<ModelKey, string> = {
  asr: "ASR",
  diarization: "Diarization",
  vision: "Vision",
  summarization: "Summarization",
  embedding: "Embeddings",
  audio_event: "Audio Events",
};

const EMPTY_SELECTIONS: ModelSelections = {
  asr: "",
  diarization: "",
  vision: "",
  summarization: "",
  embedding: "",
  audio_event: "",
};

const EMPTY_CUSTOM_MODELS: CustomModelState = {
  asr: [],
  diarization: [],
  vision: [],
  summarization: [],
  embedding: [],
  audio_event: [],
};

function mergeModelOptions(...groups: Array<ModelOption[] | undefined>): ModelOption[] {
  const merged = new Map<string, ModelOption>();
  for (const group of groups) {
    for (const option of group ?? []) {
      const key = option.name.trim().toLowerCase();
      if (!key || merged.has(key)) {
        continue;
      }
      merged.set(key, option);
    }
  }
  return Array.from(merged.values()).sort((a, b) => {
    if (a.recommended !== b.recommended) {
      return a.recommended ? -1 : 1;
    }
    return a.name.localeCompare(b.name);
  });
}

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const { refreshSession } = useAuth();
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

  const [selectedModels, setSelectedModels] = useState<ModelSelections>(EMPTY_SELECTIONS);
  const [customModels, setCustomModels] = useState<CustomModelState>(EMPTY_CUSTOM_MODELS);
  const [plannerEnabled, setPlannerEnabled] = useState(true);
  const [gpuMemoryFraction, setGpuMemoryFraction] = useState("0.75");
  const [memoryCeilingGb, setMemoryCeilingGb] = useState("");
  const [duplicatePolicy, setDuplicatePolicy] = useState("collapse_exact");
  const [duplicateExactThreshold, setDuplicateExactThreshold] = useState("0.95");
  const [duplicateProbableThreshold, setDuplicateProbableThreshold] = useState("0.85");
  const [actionItemsEnabled, setActionItemsEnabled] = useState(true);
  const [saveMessage, setSaveMessage] = useState("");
  const [modelDraftComponent, setModelDraftComponent] = useState<ModelKey>("embedding");
  const [modelDraftName, setModelDraftName] = useState("");
  const [modelVerifyMessage, setModelVerifyMessage] = useState("");
  const [verifiedCandidate, setVerifiedCandidate] = useState<{
    component: ModelKey;
    name: string;
  } | null>(null);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [passwordMessage, setPasswordMessage] = useState("");

  useEffect(() => {
    if (!settings) return;
    setSelectedModels({
      asr: settings.models.asr_model,
      diarization: settings.models.diarization_model,
      vision: settings.models.vision_model,
      summarization: settings.models.summarization_model,
      embedding: settings.models.embedding_model,
      audio_event: settings.models.audio_event_model,
    });
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
    setActionItemsEnabled(settings.action_items.enabled);
  }, [settings]);

  const modelOptions = useMemo(() => {
    const byKey = {} as Record<ModelKey, ModelOption[]>;
    for (const key of MODEL_KEYS) {
      const current = selectedModels[key]
        ? [{ name: selectedModels[key], size_gb: 0, recommended: false }]
        : [];
      byKey[key] = mergeModelOptions(
        availableModels?.[key],
        customModels[key],
        current,
      );
    }
    return byKey;
  }, [availableModels, customModels, selectedModels]);

  const saveMutation = useMutation({
    mutationFn: () =>
      api.updateSettings({
        asr_model: selectedModels.asr,
        diarization_model: selectedModels.diarization,
        vision_model: selectedModels.vision,
        summarization_model: selectedModels.summarization,
        embedding_model: selectedModels.embedding,
        audio_event_model: selectedModels.audio_event,
        runtime_planner_enabled: plannerEnabled,
        gpu_memory_fraction: Number(gpuMemoryFraction),
        runtime_memory_ceiling_gb: memoryCeilingGb === "" ? null : Number(memoryCeilingGb),
        duplicate_detection_policy: duplicatePolicy,
        duplicate_exact_threshold: Number(duplicateExactThreshold),
        duplicate_probable_threshold: Number(duplicateProbableThreshold),
        action_items_enabled: actionItemsEnabled,
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

  const verifyMutation = useMutation({
    mutationFn: ({ component, modelName }: { component: ModelKey; modelName: string }) =>
      api.verifyModel(component, modelName),
    onSuccess: (response) => {
      setModelVerifyMessage(response.detail);
      setVerifiedCandidate(
        response.exists
          ? {
              component: response.component as ModelKey,
              name: response.model_name,
            }
          : null
      );
    },
    onError: (error) => {
      setVerifiedCandidate(null);
      setModelVerifyMessage(error instanceof Error ? error.message : "Verification failed");
    },
  });

  const passwordMutation = useMutation({
    mutationFn: () => api.changePassword(currentPassword, newPassword),
    onSuccess: async () => {
      setPasswordMessage("Password updated");
      setCurrentPassword("");
      setNewPassword("");
      await refreshSession();
    },
    onError: (error) => {
      setPasswordMessage(error instanceof Error ? error.message : "Password update failed");
    },
  });

  if (settingsLoading || hardwareLoading) {
    return <div className="py-12 text-center text-slate-500 dark:text-slate-400">Loading...</div>;
  }

  const runtimePlan = settings?.runtime_planner;

  const handleModelChange = (key: ModelKey, value: string) => {
    setSelectedModels((current) => ({ ...current, [key]: value }));
  };

  const addVerifiedModel = () => {
    if (!verifiedCandidate) {
      return;
    }
    const option: ModelOption = {
      name: verifiedCandidate.name,
      size_gb: 0,
      recommended: false,
    };
    setCustomModels((current) => ({
      ...current,
      [verifiedCandidate.component]: mergeModelOptions(current[verifiedCandidate.component], [option]),
    }));
    setSelectedModels((current) => ({
      ...current,
      [verifiedCandidate.component]: verifiedCandidate.name,
    }));
    setModelVerifyMessage(`Added ${verifiedCandidate.name} to ${MODEL_LABELS[verifiedCandidate.component]}.`);
    setModelDraftName("");
    setVerifiedCandidate(null);
  };

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

        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          {MODEL_KEYS.map((key) => (
            <label
              key={key}
              className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60"
            >
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">
                {MODEL_LABELS[key]}
              </span>
              <select
                value={selectedModels[key]}
                onChange={(event) => handleModelChange(key, event.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              >
                {modelOptions[key].map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name}
                    {model.recommended ? " (recommended)" : ""}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>

        <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
          <div className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
            <ShieldCheck className="h-4 w-4" />
            Verify and add a model
          </div>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Use a built-in registry name or a Hugging Face repo id, verify it, then add it to the selector.
          </p>
          <div className="mt-3 grid gap-3 lg:grid-cols-[180px_minmax(0,1fr)_auto_auto]">
            <select
              value={modelDraftComponent}
              onChange={(event) => {
                setModelDraftComponent(event.target.value as ModelKey);
                setVerifiedCandidate(null);
                setModelVerifyMessage("");
              }}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
            >
              {MODEL_KEYS.map((key) => (
                <option key={key} value={key}>
                  {MODEL_LABELS[key]}
                </option>
              ))}
            </select>
            <input
              type="text"
              value={modelDraftName}
              onChange={(event) => {
                setModelDraftName(event.target.value);
                setVerifiedCandidate(null);
                setModelVerifyMessage("");
              }}
              placeholder="Model id or built-in registry name"
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
            />
            <button
              type="button"
              onClick={() =>
                verifyMutation.mutate({
                  component: modelDraftComponent,
                  modelName: modelDraftName.trim(),
                })
              }
              disabled={verifyMutation.isPending || modelDraftName.trim().length === 0}
              className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
            >
              {verifyMutation.isPending ? "Checking..." : "Check"}
            </button>
            <button
              type="button"
              onClick={addVerifiedModel}
              disabled={!verifiedCandidate}
              className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
            >
              Add
            </button>
          </div>
          {modelVerifyMessage ? (
            <p className="mt-3 flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
              {verifiedCandidate ? <CheckCircle2 className="h-4 w-4 text-emerald-600" /> : null}
              {modelVerifyMessage}
            </p>
          ) : null}
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Current ASR Family</p>
            <p className="mt-1 font-medium text-slate-900 dark:text-slate-100">{settings?.models.asr_family}</p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">ROCm Runtime</p>
            <p className="mt-1 font-medium capitalize text-slate-900 dark:text-slate-100">
              {settings?.models.rocm_llm_runtime.replaceAll("_", " ")}
            </p>
          </div>
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-2">
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
          <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
            <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">Todos / Action Items</span>
            <select
              value={actionItemsEnabled ? "enabled" : "disabled"}
              onChange={(event) => setActionItemsEnabled(event.target.value === "enabled")}
              className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
            >
              <option value="enabled">Enabled</option>
              <option value="disabled">Disabled</option>
            </select>
          </label>
        </div>

        {saveMessage ? (
          <p className="mt-4 text-sm text-slate-600 dark:text-slate-300">{saveMessage}</p>
        ) : null}
      </div>

      <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
        <h2 className="flex items-center text-lg font-semibold text-slate-900 dark:text-slate-100">
          <ShieldCheck className="mr-2 h-5 w-5 text-emerald-600" />
          Security
        </h2>
        <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_auto]">
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">Current Password</span>
              <input
                type="password"
                value={currentPassword}
                onChange={(event) => setCurrentPassword(event.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              />
            </label>
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800/60">
              <span className="mb-1 block text-sm text-slate-600 dark:text-slate-300">New Password</span>
              <input
                type="password"
                minLength={12}
                value={newPassword}
                onChange={(event) => setNewPassword(event.target.value)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              />
            </label>
          </div>
          <button
            type="button"
            onClick={() => {
              setPasswordMessage("");
              passwordMutation.mutate();
            }}
            disabled={passwordMutation.isPending || currentPassword.length === 0 || newPassword.length < 12}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900"
          >
            {passwordMutation.isPending ? "Updating..." : "Change Password"}
          </button>
        </div>
        {passwordMessage ? (
          <p className="mt-4 text-sm text-slate-600 dark:text-slate-300">{passwordMessage}</p>
        ) : null}
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
            <p className="text-sm text-slate-500 dark:text-slate-400">Free Budget</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {runtimePlan?.effective_memory_budget_gb} GB / {runtimePlan?.available_accelerator_memory_gb} GB free
            </p>
          </div>
          <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800/60">
            <p className="text-sm text-slate-500 dark:text-slate-400">Installed VRAM</p>
            <p className="text-lg font-medium text-slate-900 dark:text-slate-100">
              {runtimePlan?.total_accelerator_memory_gb} GB
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
              {runtimePlan?.preferred_model_devices && Object.keys(runtimePlan.preferred_model_devices).length
                ? Object.entries(runtimePlan.preferred_model_devices).map(([key, devices]) => (
                    <p key={key}>
                      {key} placement: {devices.join(", ")}
                    </p>
                  ))
                : null}
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
