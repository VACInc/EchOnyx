"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  CheckCircle2,
  ChevronDown,
  Download,
  HardDrive,
  Info,
  KeyRound,
  Plus,
  Save,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Zap,
} from "lucide-react";

import { useAuth } from "@/components/auth-gate";
import { SectionCard } from "@/components/settings/section-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { ErrorState } from "@/components/ui/error-state";
import { Field } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { StatusBadge } from "@/components/ui/status-badge";
import { useToast } from "@/components/ui/toast";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";

const MODEL_KEYS = [
  "asr",
  "diarization",
  "vision",
  "summarization",
  "embedding",
  "audio_event",
] as const;

type ModelKey = (typeof MODEL_KEYS)[number];
type ModelRecommendations = Awaited<ReturnType<typeof api.getModelRecommendations>>;
type ModelRecommendationEntry = ModelRecommendations["recommendations"][string];
type ModelStatus = Awaited<ReturnType<typeof api.getModelStatus>>;

type ModelOption = {
  name: string;
  size_gb: number;
  recommended: boolean;
};

type ModelSelections = Record<ModelKey, string>;
type CustomModelState = Record<ModelKey, ModelOption[]>;

type DownloadStatus = {
  error?: string | null;
  eta_seconds?: number | null;
  expected_size_gb?: number | null;
  file_size_gb?: number | null;
  model_name: string;
  progress_percent?: number | null;
  speed_mbps?: number | null;
  status: string;
};

type DownloadRow = {
  component: ModelKey;
  expectedSizeGb: number | null;
  id: string;
  modelName: string;
  reason?: string;
  roles: Array<"Configured" | "Recommended">;
};

const MODEL_LABELS: Record<ModelKey, string> = {
  asr: "ASR",
  diarization: "Diarization",
  vision: "Vision",
  summarization: "Summarization",
  embedding: "Embeddings",
  audio_event: "Audio Events",
};

const MODEL_DESCRIPTIONS: Record<ModelKey, string> = {
  asr: "Turns spoken audio into transcript text for the rest of the pipeline.",
  diarization: "Separates transcript segments by speaker where the model supports it.",
  vision: "Describes slides and visual frames for summaries and search context.",
  summarization: "Creates summaries, decisions, topics, and extracted action items.",
  embedding: "Indexes transcripts and summaries for semantic search and similarity.",
  audio_event: "Detects non-speech audio cues that can improve media understanding.",
};

const MODEL_STATUS_KEYS: Record<ModelKey, string> = {
  asr: "whisper",
  diarization: "diarization",
  vision: "vision",
  summarization: "summarization",
  embedding: "embedding",
  audio_event: "audio_event",
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

const DUPLICATE_POLICY_DESCRIPTIONS: Record<string, string> = {
  off: "Do not classify duplicates or suppress repeated uploads.",
  warn: "Mark likely duplicates but keep every upload visible.",
  collapse_exact: "Hide uploads only when they cross the exact-match threshold.",
  collapse_probable: "Hide uploads that cross either the probable or exact threshold.",
};

function mergeModelOptions(...groups: Array<ModelOption[] | undefined>): ModelOption[] {
  const merged = new Map<string, ModelOption>();
  for (const group of groups) {
    for (const option of group ?? []) {
      const key = option.name.trim().toLowerCase();
      if (!key || merged.has(key)) continue;
      merged.set(key, option);
    }
  }
  return Array.from(merged.values()).sort((a, b) => {
    if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function formatGb(value: number | null | undefined, digits = 1): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "Unknown";
  return `${value.toFixed(digits)} GB`;
}

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "0%";
  return `${Math.round(value)}%`;
}

function formatEta(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || Number.isNaN(seconds)) return "ETA unknown";
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s left`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s left`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m left`;
}

function titleCase(value: string): string {
  return value
    .split(" ")
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function cleanLabel(value: string): string {
  return value.replaceAll("_", " ").replaceAll("-", " ").trim();
}

function humanizeValue(value: string | null | undefined): string {
  if (!value) return "Unknown";
  const trimmed = value.trim();
  const parenthetical = trimmed.match(/^(.+?)\s*\((.+)\)$/);
  if (parenthetical) {
    return `${titleCase(cleanLabel(parenthetical[1]))} — ${cleanLabel(parenthetical[2]).toLowerCase()}`;
  }
  return titleCase(cleanLabel(trimmed));
}

function validationRangeError(
  value: string,
  label: string,
  min: number,
  max: number,
): string | undefined {
  if (value.trim() === "") return `${label} is required.`;
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return `${label} must be a number.`;
  if (numeric < min || numeric > max) return `${label} must be between ${min} and ${max}.`;
  return undefined;
}

function buildDownloadRows(
  selectedModels: ModelSelections,
  recommendations: ModelRecommendations | undefined,
  availableModels: Partial<Record<ModelKey, ModelOption[]>> | undefined,
): DownloadRow[] {
  const rows = new Map<string, DownloadRow>();

  const addRow = (
    component: ModelKey,
    modelName: string | undefined,
    role: "Configured" | "Recommended",
    recommendation?: ModelRecommendationEntry,
  ) => {
    const cleanName = modelName?.trim();
    if (!cleanName) return;
    const id = `${component}:${cleanName.toLowerCase()}`;
    const availableSize = availableModels?.[component]?.find((model) => model.name === cleanName)?.size_gb;
    const existing = rows.get(id);
    if (existing) {
      if (!existing.roles.includes(role)) existing.roles.push(role);
      existing.reason = existing.reason ?? recommendation?.reason;
      existing.expectedSizeGb = existing.expectedSizeGb ?? recommendation?.expected_size_gb ?? availableSize ?? null;
      return;
    }
    rows.set(id, {
      component,
      expectedSizeGb: recommendation?.expected_size_gb ?? availableSize ?? null,
      id,
      modelName: cleanName,
      reason: recommendation?.reason,
      roles: [role],
    });
  };

  for (const component of MODEL_KEYS) {
    const recommendation = recommendations?.recommendations[component];
    addRow(component, selectedModels[component], "Configured");
    addRow(component, recommendation?.model_name, "Recommended", recommendation);
  }

  return Array.from(rows.values()).sort((a, b) => {
    const componentOrder = MODEL_KEYS.indexOf(a.component) - MODEL_KEYS.indexOf(b.component);
    if (componentOrder !== 0) return componentOrder;
    return a.modelName.localeCompare(b.modelName);
  });
}

function statusForRow(
  row: DownloadRow,
  modelStatus: ModelStatus | undefined,
  recommendations: ModelRecommendations | undefined,
): DownloadStatus {
  const activeDownload = modelStatus?.active_downloads.find((download) => download.model_name === row.modelName);
  if (activeDownload) {
    return { ...activeDownload, status: "downloading" };
  }

  const configuredStatus = modelStatus?.models[MODEL_STATUS_KEYS[row.component]];
  if (configuredStatus?.model_name === row.modelName) {
    return configuredStatus;
  }

  const recommendation = recommendations?.recommendations[row.component];
  if (recommendation?.model_name === row.modelName) {
    return {
      expected_size_gb: recommendation.expected_size_gb,
      model_name: row.modelName,
      status: recommendation.cached ? "cached" : "uncached",
    };
  }

  return {
    expected_size_gb: row.expectedSizeGb,
    model_name: row.modelName,
    status: "uncached",
  };
}

function SectionDisclosure({
  children,
  defaultOpen = true,
  label,
}: {
  children: React.ReactNode;
  defaultOpen?: boolean;
  label: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details
      className="rounded-lg border border-border bg-muted/30"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
        {label}
        <ChevronDown className={cn("h-4 w-4 transition", open && "rotate-180")} aria-hidden="true" />
      </summary>
      <div className="border-t border-border p-4">{children}</div>
    </details>
  );
}

function KeyValueGrid({ items }: { items: Array<{ label: string; value: React.ReactNode }> }) {
  return (
    <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {items.map((item) => (
        <div key={item.label} className="rounded-lg border border-border bg-card p-3">
          <dt className="text-xs text-muted-foreground">{item.label}</dt>
          <dd className="mt-1 break-words text-sm font-medium text-card-foreground">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function SettingsSkeleton() {
  return (
    <div className="space-y-6">
      {[0, 1, 2, 3, 4, 5, 6].map((section) => (
        <Card key={section} className="p-5 sm:p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1 space-y-2">
              <Skeleton className="h-6 w-48" />
              <Skeleton className="h-4 w-full max-w-xl" />
            </div>
            <Skeleton className="h-10 w-32 rounded-lg" />
          </div>
          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
        </Card>
      ))}
    </div>
  );
}

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const confirm = useConfirm();
  const toast = useToast();
  const { refreshSession, session } = useAuth();
  const [pollDownloads, setPollDownloads] = useState(false);
  const [hadActiveDownloads, setHadActiveDownloads] = useState(false);

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });
  const hardwareQuery = useQuery({
    queryKey: ["hardware"],
    queryFn: api.getHardwareInfo,
  });
  const availableModelsQuery = useQuery({
    queryKey: ["available-models"],
    queryFn: api.getAvailableModels,
  });
  const recommendationsQuery = useQuery({
    queryKey: ["model-recommendations"],
    queryFn: api.getModelRecommendations,
  });
  const modelStatusQuery = useQuery({
    queryKey: ["model-status"],
    queryFn: api.getModelStatus,
    refetchInterval: pollDownloads ? 2000 : false,
  });

  const settings = settingsQuery.data;
  const hardware = hardwareQuery.data;
  const recommendations = recommendationsQuery.data;
  const modelStatus = modelStatusQuery.data;

  const [selectedModels, setSelectedModels] = useState<ModelSelections>(EMPTY_SELECTIONS);
  const [customModels, setCustomModels] = useState<CustomModelState>(EMPTY_CUSTOM_MODELS);
  const [plannerEnabled, setPlannerEnabled] = useState(true);
  const [gpuMemoryFraction, setGpuMemoryFraction] = useState("0.75");
  const [memoryCeilingGb, setMemoryCeilingGb] = useState("");
  const [duplicatePolicy, setDuplicatePolicy] = useState("collapse_exact");
  const [duplicateExactThreshold, setDuplicateExactThreshold] = useState("0.95");
  const [duplicateProbableThreshold, setDuplicateProbableThreshold] = useState("0.85");
  const [actionItemsEnabled, setActionItemsEnabled] = useState(true);
  const [modelDraftComponent, setModelDraftComponent] = useState<ModelKey>("embedding");
  const [modelDraftName, setModelDraftName] = useState("");
  const [modelVerifyMessage, setModelVerifyMessage] = useState("");
  const [verifiedCandidate, setVerifiedCandidate] = useState<{
    component: ModelKey;
    name: string;
  } | null>(null);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [downloadRequestId, setDownloadRequestId] = useState<string | null>(null);

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
        : settings.runtime_planner.memory_ceiling_gb.toString(),
    );
    setDuplicatePolicy(settings.duplicates.policy);
    setDuplicateExactThreshold(settings.duplicates.exact_threshold.toString());
    setDuplicateProbableThreshold(settings.duplicates.probable_threshold.toString());
    setActionItemsEnabled(settings.action_items.enabled);
  }, [settings]);

  useEffect(() => {
    const hasActiveDownloads = modelStatus?.active_downloads.some((download) => download.status === "downloading");
    setPollDownloads(Boolean(hasActiveDownloads));
    if (hasActiveDownloads) {
      setHadActiveDownloads(true);
      return;
    }
    if (modelStatus && hadActiveDownloads) {
      setHadActiveDownloads(false);
      void queryClient.invalidateQueries({ queryKey: ["model-recommendations"] });
    }
  }, [hadActiveDownloads, modelStatus, queryClient]);

  const modelOptions = useMemo(() => {
    const byKey = {} as Record<ModelKey, ModelOption[]>;
    for (const key of MODEL_KEYS) {
      const current = selectedModels[key]
        ? [{ name: selectedModels[key], size_gb: 0, recommended: false }]
        : [];
      byKey[key] = mergeModelOptions(availableModelsQuery.data?.[key], customModels[key], current);
    }
    return byKey;
  }, [availableModelsQuery.data, customModels, selectedModels]);

  const downloadRows = useMemo(
    () => buildDownloadRows(selectedModels, recommendations, availableModelsQuery.data),
    [availableModelsQuery.data, recommendations, selectedModels],
  );

  const gpuMemoryFractionError = validationRangeError(gpuMemoryFraction, "GPU memory fraction", 0.1, 1);
  const memoryCeilingError =
    memoryCeilingGb.trim() === "" || Number(memoryCeilingGb) >= 0
      ? undefined
      : "Memory ceiling must be 0 GB or greater.";
  const duplicateExactError = validationRangeError(duplicateExactThreshold, "Exact threshold", 0, 1);
  const duplicateProbableRangeError = validationRangeError(duplicateProbableThreshold, "Probable threshold", 0, 1);
  const duplicateProbableOrderError =
    !duplicateExactError &&
    !duplicateProbableRangeError &&
    Number(duplicateProbableThreshold) > Number(duplicateExactThreshold)
      ? "Probable threshold must be less than or equal to the exact threshold."
      : undefined;
  const duplicateProbableError = duplicateProbableRangeError ?? duplicateProbableOrderError;
  const hasSettingsValidationErrors = Boolean(
    gpuMemoryFractionError || memoryCeilingError || duplicateExactError || duplicateProbableError,
  );
  const newPasswordError =
    newPassword.length > 0 && newPassword.length < 12 ? "Use at least 12 characters." : undefined;

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
      toast({
        title: "Settings saved",
        description: "Model and runtime changes will apply the next time those services load.",
        variant: "success",
      });
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["settings"] }),
        queryClient.invalidateQueries({ queryKey: ["hardware"] }),
        queryClient.invalidateQueries({ queryKey: ["model-recommendations"] }),
        queryClient.invalidateQueries({ queryKey: ["model-status"] }),
      ]);
    },
    onError: (error) => {
      toast({
        title: "Settings were not saved",
        description: getErrorMessage(error, "Review the fields and try saving again."),
        variant: "error",
      });
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
          : null,
      );
    },
    onError: (error) => {
      setVerifiedCandidate(null);
      setModelVerifyMessage(getErrorMessage(error, "Verification failed."));
    },
  });

  const downloadMutation = useMutation({
    mutationFn: ({ component, modelName }: { component: ModelKey; modelName: string }) =>
      api.downloadModel(component, modelName),
    onMutate: ({ component, modelName }) => {
      setDownloadRequestId(`${component}:${modelName}`);
    },
    onSuccess: async (response) => {
      const companions = response.companions ?? [];
      const failedCompanions = companions.filter((companion) => companion.status === "error");
      const companionDownloading = companions.some((companion) => companion.status === "downloading");
      setPollDownloads(response.status === "downloading" || companionDownloading);
      if (failedCompanions.length > 0) {
        toast({
          title: "Companion download failed",
          description: failedCompanions
            .map((companion) => `${companion.model_name}: ${companion.detail ?? "unknown error"}`)
            .join("; "),
          variant: "error",
        });
      } else {
        toast({
          title:
            response.status === "cached" && !companionDownloading
              ? "Model already cached"
              : "Download started",
          description: response.note ?? response.model_name,
          variant: "success",
        });
      }
      await queryClient.invalidateQueries({ queryKey: ["model-status"] });
      await queryClient.invalidateQueries({ queryKey: ["model-recommendations"] });
    },
    onError: (error) => {
      toast({
        title: "Download could not start",
        description: getErrorMessage(error, "Check the model name, license access, and disk space."),
        variant: "error",
      });
    },
    onSettled: () => {
      setDownloadRequestId(null);
    },
  });

  const passwordMutation = useMutation({
    mutationFn: () => api.changePassword(currentPassword, newPassword),
    onSuccess: async () => {
      toast({
        title: "Password updated",
        description: "Use the new password the next time you sign in.",
        variant: "success",
      });
      setCurrentPassword("");
      setNewPassword("");
      await refreshSession();
    },
    onError: (error) => {
      toast({
        title: "Password was not updated",
        description: getErrorMessage(error, "Check the current password and try again."),
        variant: "error",
      });
    },
  });

  if (settingsQuery.isLoading || hardwareQuery.isLoading) {
    return <SettingsSkeleton />;
  }

  if (settingsQuery.isError || hardwareQuery.isError || !settings || !hardware) {
    return (
      <ErrorState
        title="Settings failed to load"
        message={getErrorMessage(
          settingsQuery.error ?? hardwareQuery.error,
          "Refresh to try loading settings and hardware information again.",
        )}
        onRetry={() => {
          void settingsQuery.refetch();
          void hardwareQuery.refetch();
        }}
      />
    );
  }

  const runtimePlan = settings.runtime_planner;
  const recommendationsExceedDisk =
    recommendations &&
    recommendations.total_additional_download_gb > recommendations.free_disk_gb;

  const handleModelChange = (key: ModelKey, value: string) => {
    setSelectedModels((current) => ({ ...current, [key]: value }));
  };

  const handleSaveSettings = () => {
    if (hasSettingsValidationErrors) {
      toast({
        title: "Fix validation errors",
        description: "Review the highlighted runtime and duplicate thresholds before saving.",
        variant: "error",
      });
      return;
    }
    saveMutation.mutate();
  };

  const addVerifiedModel = () => {
    if (!verifiedCandidate) return;
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

  const confirmDownload = async (row: DownloadRow, status: DownloadStatus) => {
    const expectedSizeGb = status.expected_size_gb ?? row.expectedSizeGb;
    const confirmed = await confirm({
      title: `Download ${MODEL_LABELS[row.component]} model?`,
      description: (
        <span>
          Expected size: {formatGb(expectedSizeGb)}. Free disk: {formatGb(recommendations?.free_disk_gb)}.
          Recommendation set needs {formatGb(recommendations?.total_additional_download_gb)} total additional downloads.
        </span>
      ),
      confirmLabel: "Download",
    });
    if (!confirmed) return;
    downloadMutation.mutate({ component: row.component, modelName: row.modelName });
  };

  const processingItems = [
    { label: "Max video length", value: `${settings.processing.max_video_length_hours} hours` },
    { label: "Keyframe interval", value: `${settings.processing.keyframe_extraction_interval} seconds` },
    { label: "Frame persistence", value: `${settings.processing.frame_persistence_seconds} seconds` },
    { label: "Frame change threshold", value: settings.processing.frame_change_threshold },
    { label: "Frame stability threshold", value: settings.processing.frame_stability_threshold },
    { label: "Frame dedupe threshold", value: settings.processing.frame_dedupe_threshold },
    { label: "Frame resize width", value: `${settings.processing.frame_resize_width}px` },
    { label: "Max keyframes", value: settings.processing.max_keyframes },
    { label: "Min speech duration", value: `${settings.processing.min_speech_duration} seconds` },
    { label: "Batch concurrent jobs", value: settings.processing.batch_concurrent_jobs },
    { label: "Summary chunk length", value: `${settings.processing.summary_chunk_minutes} minutes` },
    { label: "Summary chunk overlap", value: `${settings.processing.summary_chunk_overlap_minutes} minutes` },
    { label: "Detected backend", value: hardware.active_backend },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Settings</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Configure local models, runtime planning, duplicate handling, and account access.
          </p>
        </div>
        <Button type="button" onClick={handleSaveSettings} loading={saveMutation.isPending}>
          <Save className="h-4 w-4" aria-hidden="true" />
          Save Settings
        </Button>
      </div>

      <SectionCard
        title="Models"
        description="Controls which model each processing stage uses when work is queued or loaded."
        icon={Zap}
      >
        <div className="rounded-lg border border-info/25 bg-info/10 p-3 text-sm text-foreground">
          Changing a model may require a download and a model reload before processing uses the new selection.
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          {MODEL_KEYS.map((key) => (
            <Field key={key} label={MODEL_LABELS[key]} description={MODEL_DESCRIPTIONS[key]}>
              <Select value={selectedModels[key]} onChange={(event) => handleModelChange(key, event.target.value)}>
                {modelOptions[key].map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name}
                    {model.recommended ? " (recommended)" : ""}
                    {model.size_gb ? ` - ${formatGb(model.size_gb)}` : ""}
                  </option>
                ))}
              </Select>
            </Field>
          ))}
        </div>

        <div className="mt-5 rounded-lg border border-border bg-muted/30 p-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-card-foreground">Verify and Add a Model</h3>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Use a built-in registry name or Hugging Face repo id, verify it, then add it to the selector.
          </p>
          <div className="mt-4 grid gap-3 lg:grid-cols-[180px_minmax(0,1fr)_auto_auto]">
            <Select
              aria-label="Model component"
              value={modelDraftComponent}
              onChange={(event) => {
                setModelDraftComponent(event.target.value as ModelKey);
                setVerifiedCandidate(null);
                setModelVerifyMessage("");
              }}
            >
              {MODEL_KEYS.map((key) => (
                <option key={key} value={key}>
                  {MODEL_LABELS[key]}
                </option>
              ))}
            </Select>
            <Input
              type="text"
              value={modelDraftName}
              onChange={(event) => {
                setModelDraftName(event.target.value);
                setVerifiedCandidate(null);
                setModelVerifyMessage("");
              }}
              placeholder="Model id or built-in registry name"
              aria-label="Model id or built-in registry name"
            />
            <Button
              type="button"
              variant="outline"
              onClick={() =>
                verifyMutation.mutate({
                  component: modelDraftComponent,
                  modelName: modelDraftName.trim(),
                })
              }
              disabled={modelDraftName.trim().length === 0}
              loading={verifyMutation.isPending}
            >
              <Check className="h-4 w-4" aria-hidden="true" />
              Check
            </Button>
            <Button type="button" onClick={addVerifiedModel} disabled={!verifiedCandidate}>
              <Plus className="h-4 w-4" aria-hidden="true" />
              Add
            </Button>
          </div>
          {modelVerifyMessage ? (
            <p className="mt-3 flex items-start gap-2 text-sm text-muted-foreground" aria-live="polite">
              {verifiedCandidate ? (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" aria-hidden="true" />
              ) : null}
              <span>{modelVerifyMessage}</span>
            </p>
          ) : null}
        </div>

        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg border border-border bg-muted/30 p-3">
            <p className="text-xs text-muted-foreground">Current ASR family</p>
            <p className="mt-1 text-sm font-medium text-card-foreground">{humanizeValue(settings.models.asr_family)}</p>
          </div>
          <div className="rounded-lg border border-border bg-muted/30 p-3">
            <p className="text-xs text-muted-foreground">ROCm runtime</p>
            <p className="mt-1 text-sm font-medium text-card-foreground">
              {humanizeValue(settings.models.rocm_llm_runtime)}
            </p>
          </div>
        </div>
      </SectionCard>

      <SectionCard
        title="Model Downloads"
        description="Shows whether configured and recommended models are cached locally or need download."
        icon={Download}
      >
        {recommendationsExceedDisk ? (
          <div className="mb-4 rounded-lg border border-warning/25 bg-warning/10 p-3 text-sm text-foreground">
            Recommended downloads need {formatGb(recommendations.total_additional_download_gb)}, but only{" "}
            {formatGb(recommendations.free_disk_gb)} is free.
          </div>
        ) : null}

        {recommendationsQuery.isLoading || modelStatusQuery.isLoading ? (
          <div className="grid gap-3">
            <Skeleton className="h-28" />
            <Skeleton className="h-28" />
            <Skeleton className="h-28" />
          </div>
        ) : recommendationsQuery.isError || modelStatusQuery.isError ? (
          <ErrorState
            title="Model download status failed to load"
            message={getErrorMessage(
              recommendationsQuery.error ?? modelStatusQuery.error,
              "Refresh to check cache and download status again.",
            )}
            onRetry={() => {
              void recommendationsQuery.refetch();
              void modelStatusQuery.refetch();
            }}
          />
        ) : (
          <div className="grid gap-3">
            {downloadRows.map((row) => {
              const status = statusForRow(row, modelStatus, recommendations);
              const isDownloading = status.status === "downloading";
              const canDownload = status.status === "uncached";
              const progress = typeof status.progress_percent === "number" ? status.progress_percent : 0;
              const expectedSize = status.file_size_gb ?? status.expected_size_gb ?? row.expectedSizeGb;
              const requestId = `${row.component}:${row.modelName}`;

              return (
                <div key={row.id} className="rounded-lg border border-border bg-muted/30 p-4">
                  <div className="grid gap-4 lg:grid-cols-[1.1fr_0.45fr_0.7fr_auto] lg:items-start">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="break-words text-sm font-semibold text-card-foreground">{row.modelName}</h3>
                        {row.roles.map((role) => (
                          <Badge key={role} variant={role === "Recommended" ? "info" : "outline"}>
                            {role}
                          </Badge>
                        ))}
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">{MODEL_LABELS[row.component]}</p>
                      {row.reason ? <p className="mt-2 text-sm text-muted-foreground">{row.reason}</p> : null}
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Expected size</p>
                      <p className="mt-1 text-sm font-medium text-card-foreground">{formatGb(expectedSize)}</p>
                    </div>
                    <div>
                      <StatusBadge status={status.status} tabIndex={0} />
                      {isDownloading ? (
                        <div className="mt-3 space-y-2">
                          <Progress value={progress} />
                          <p className="text-xs text-muted-foreground">
                            {formatPercent(progress)} · {status.speed_mbps?.toFixed(1) ?? "0.0"} MB/s ·{" "}
                            {formatEta(status.eta_seconds)}
                          </p>
                        </div>
                      ) : status.error ? (
                        <p className="mt-2 text-xs text-destructive">{status.error}</p>
                      ) : null}
                    </div>
                    <Button
                      type="button"
                      variant="outline"
                      className="w-full lg:w-auto"
                      disabled={!canDownload}
                      loading={downloadMutation.isPending && downloadRequestId === requestId}
                      onClick={() => void confirmDownload(row, status)}
                    >
                      <Download className="h-4 w-4" aria-hidden="true" />
                      Download
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </SectionCard>

      <SectionCard
        title="Runtime"
        description="Sets how aggressively EchOnyx budgets accelerator memory for loaded models."
        icon={SlidersHorizontal}
      >
        <div className="grid gap-4 lg:grid-cols-3">
          <Field
            label="Runtime planner"
            description="When enabled, the planner chooses model residency and endpoint behavior from detected memory."
          >
            <Select
              value={plannerEnabled ? "enabled" : "disabled"}
              onChange={(event) => setPlannerEnabled(event.target.value === "enabled")}
            >
              <option value="enabled">Enabled</option>
              <option value="disabled">Disabled</option>
            </Select>
          </Field>
          <Field
            label="GPU memory fraction"
            description="Fraction of free accelerator memory the planner may budget for resident models."
            error={gpuMemoryFractionError}
          >
            <Input
              type="number"
              min="0.1"
              max="1"
              step="0.05"
              value={gpuMemoryFraction}
              onChange={(event) => setGpuMemoryFraction(event.target.value)}
            />
          </Field>
          <Field
            label="Memory ceiling"
            description="Optional GB cap for the planner. Leave blank to use detected free memory."
            error={memoryCeilingError}
          >
            <Input
              type="number"
              min="0"
              step="1"
              placeholder="Auto"
              value={memoryCeilingGb}
              onChange={(event) => setMemoryCeilingGb(event.target.value)}
            />
          </Field>
        </div>
      </SectionCard>

      <SectionCard
        title="Duplicates & Features"
        description="Controls duplicate suppression and whether generated action items appear in the app."
        icon={Info}
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <Field label="Duplicate policy" description={DUPLICATE_POLICY_DESCRIPTIONS[duplicatePolicy]}>
            <Select value={duplicatePolicy} onChange={(event) => setDuplicatePolicy(event.target.value)}>
              <option value="off">Off</option>
              <option value="warn">Warn only</option>
              <option value="collapse_exact">Collapse exact duplicates</option>
              <option value="collapse_probable">Collapse probable duplicates</option>
            </Select>
          </Field>
          <Field
            label="Todos / Action Items"
            description="Disabling hides generated action items and removes the Todos navigation entry."
          >
            <button
              type="button"
              role="switch"
              aria-checked={actionItemsEnabled}
              className="inline-flex h-10 w-full items-center justify-between gap-3 rounded-lg border border-input bg-card px-3 text-sm text-card-foreground shadow-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              onClick={() => setActionItemsEnabled((current) => !current)}
            >
              <span>{actionItemsEnabled ? "Enabled" : "Disabled"}</span>
              <span
                className={cn(
                  "flex h-5 w-9 items-center rounded-full border border-border p-0.5 transition",
                  actionItemsEnabled ? "justify-end bg-primary" : "justify-start bg-muted",
                )}
                aria-hidden="true"
              >
                <span className="h-4 w-4 rounded-full bg-background shadow-sm" />
              </span>
            </button>
          </Field>
          <Field
            label="Exact threshold"
            description="Similarity score required before uploads are treated as exact duplicates."
            error={duplicateExactError}
          >
            <Input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={duplicateExactThreshold}
              onChange={(event) => setDuplicateExactThreshold(event.target.value)}
            />
          </Field>
          <Field
            label="Probable threshold"
            description="Similarity score for likely duplicates. It cannot be higher than the exact threshold."
            error={duplicateProbableError}
          >
            <Input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={duplicateProbableThreshold}
              onChange={(event) => setDuplicateProbableThreshold(event.target.value)}
            />
          </Field>
        </div>
      </SectionCard>

      <SectionCard
        title="Security"
        description="Manages local password access when password authentication is enabled."
        icon={KeyRound}
      >
        {session.password_enabled ? (
          <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-start">
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Current password" description="Required before setting a new local password.">
                <Input
                  type="password"
                  value={currentPassword}
                  onChange={(event) => setCurrentPassword(event.target.value)}
                  autoComplete="current-password"
                />
              </Field>
              <Field label="New password" description="Use at least 12 characters." error={newPasswordError}>
                <Input
                  type="password"
                  minLength={12}
                  value={newPassword}
                  onChange={(event) => setNewPassword(event.target.value)}
                  autoComplete="new-password"
                />
              </Field>
            </div>
            <Button
              type="button"
              onClick={() => passwordMutation.mutate()}
              disabled={currentPassword.length === 0 || newPassword.length < 12}
              loading={passwordMutation.isPending}
            >
              <ShieldCheck className="h-4 w-4" aria-hidden="true" />
              Change Password
            </Button>
          </div>
        ) : (
          <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
            Local password auth is disabled. Sign in through the configured identity provider.
          </div>
        )}
      </SectionCard>

      <SectionCard
        title="Runtime Plan"
        description="Shows the active planner decision for model placement, residency, and endpoint loading."
        icon={Settings}
      >
        <SectionDisclosure label="Planner decision">
          <div className="grid gap-4">
            <KeyValueGrid
              items={[
                { label: "Placement mode", value: humanizeValue(runtimePlan.placement_mode) },
                {
                  label: "Budget vs installed memory",
                  value: `${formatGb(runtimePlan.effective_memory_budget_gb)} budget / ${formatGb(runtimePlan.total_accelerator_memory_gb)} installed`,
                },
                { label: "Available memory", value: formatGb(runtimePlan.available_accelerator_memory_gb) },
                { label: "Accelerators", value: runtimePlan.accelerator_count },
                {
                  label: "Worker loading",
                  value: `${humanizeValue(runtimePlan.worker_model_loading)} — ${cleanLabel(runtimePlan.worker_execution_mode).toLowerCase()}`,
                },
                { label: "Endpoint loading", value: humanizeValue(runtimePlan.endpoint_model_loading) },
                {
                  label: "Endpoint idle timeout",
                  value: `${runtimePlan.endpoint_idle_timeout_recommendation_s} seconds`,
                },
                {
                  label: "Shutdown endpoint after request",
                  value: runtimePlan.shutdown_endpoint_after_request ? "Yes" : "No",
                },
              ]}
            />

            <div className="rounded-lg border border-border bg-card p-4">
              <h3 className="text-sm font-semibold text-card-foreground">Resident models</h3>
              <div className="mt-3 flex flex-wrap gap-2">
                {runtimePlan.keep_resident_models.length ? (
                  runtimePlan.keep_resident_models.map((model) => (
                    <Badge key={model} variant="secondary">
                      {humanizeValue(model)}
                    </Badge>
                  ))
                ) : (
                  <p className="text-sm text-muted-foreground">No resident worker models planned.</p>
                )}
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="text-sm font-semibold text-card-foreground">Per-model placement</h3>
                <dl className="mt-3 divide-y divide-border text-sm">
                  {Object.keys(runtimePlan.preferred_model_devices).length ? (
                    Object.entries(runtimePlan.preferred_model_devices).map(([model, devices]) => (
                      <div key={model} className="grid gap-1 py-2 sm:grid-cols-[0.8fr_1.2fr]">
                        <dt className="font-medium text-card-foreground">{humanizeValue(model)}</dt>
                        <dd className="text-muted-foreground">{devices.length ? devices.join(", ") : "Auto"}</dd>
                      </div>
                    ))
                  ) : (
                    <div className="py-2 text-muted-foreground">No model-specific placement overrides.</div>
                  )}
                  <div className="grid gap-1 py-2 sm:grid-cols-[0.8fr_1.2fr]">
                    <dt className="font-medium text-card-foreground">Worker devices</dt>
                    <dd className="text-muted-foreground">
                      {runtimePlan.preferred_worker_devices.length
                        ? runtimePlan.preferred_worker_devices.join(", ")
                        : "Auto"}
                    </dd>
                  </div>
                  <div className="grid gap-1 py-2 sm:grid-cols-[0.8fr_1.2fr]">
                    <dt className="font-medium text-card-foreground">Endpoint devices</dt>
                    <dd className="text-muted-foreground">
                      {runtimePlan.preferred_endpoint_devices.length
                        ? runtimePlan.preferred_endpoint_devices.join(", ")
                        : "Auto"}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="rounded-lg border border-border bg-card p-4">
                <h3 className="text-sm font-semibold text-card-foreground">Estimated memory by model</h3>
                <dl className="mt-3 divide-y divide-border text-sm">
                  {Object.entries(runtimePlan.estimated_memory_by_model_gb).map(([model, memory]) => (
                    <div key={model} className="grid gap-1 py-2 sm:grid-cols-[0.8fr_1.2fr]">
                      <dt className="font-medium text-card-foreground">{humanizeValue(model)}</dt>
                      <dd className="text-muted-foreground">{formatGb(memory)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>

            <div className="rounded-lg border border-info/25 bg-info/10 p-4">
              <div className="flex items-start gap-3">
                <Info className="mt-0.5 h-5 w-5 shrink-0 text-info" aria-hidden="true" />
                <div>
                  <h3 className="text-sm font-semibold text-foreground">Planner notes</h3>
                  <div className="mt-2 space-y-2 text-sm text-muted-foreground">
                    {runtimePlan.notes.length ? (
                      runtimePlan.notes.map((note) => <p key={note}>{note}</p>)
                    ) : (
                      <p>No planner warnings.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </SectionDisclosure>
      </SectionCard>

      <SectionCard
        title="Processing"
        description="Displays processing limits for upload length, frame extraction, speech chunks, and batch work."
        icon={HardDrive}
      >
        <SectionDisclosure label="Processing limits">
          <KeyValueGrid items={processingItems} />
        </SectionDisclosure>
      </SectionCard>
    </div>
  );
}
