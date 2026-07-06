"use client";

import { useCallback, useMemo, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  CheckCircle2,
  FileVideo,
  Info,
  RotateCcw,
  Trash2,
  Upload,
  XCircle,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/components/ui/toast";
import { cn } from "@/lib/utils";
import { api, type UploadProgressEvent } from "@/lib/api";

const ACCEPTED_EXTENSIONS = [".mp4", ".webm", ".mov", ".avi", ".mkv"];
const ACCEPTED_FORMATS = "MP4, WebM, MOV, AVI, MKV";
const BYTES_PER_GB = 1024 * 1024 * 1024;
const CLOSE_DELAY_MS = 1000;

type SettingsResponse = Awaited<ReturnType<typeof api.getSettings>>;
type BatchResponse = Awaited<ReturnType<typeof api.createBatch>>;
type UploadStatus = "ready" | "uploading" | "success" | "error";

type SelectedFile = {
  file: File;
  id: string;
  message?: string;
  progress: number;
  status: UploadStatus;
};

type UploadDropzoneProps = {
  onUploaded?: () => void;
};

function createFileId(file: File) {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2)}`;
}

function readNumericField(source: unknown, key: string): number | null {
  if (!source || typeof source !== "object") return null;
  const value = (source as Record<string, unknown>)[key];
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function getKnownMaxUploadSizeGb(settings: SettingsResponse | undefined): number | null {
  const processing = settings?.processing;
  return (
    readNumericField(settings, "max_upload_size_gb") ??
    readNumericField(settings, "maxUploadSizeGb") ??
    readNumericField(processing, "max_upload_size_gb") ??
    readNumericField(processing, "maxUploadSizeGb")
  );
}

function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < BYTES_PER_GB) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / BYTES_PER_GB).toFixed(2)} GB`;
}

function formatUploadCap(gb: number) {
  return Number.isInteger(gb) ? `${gb} GB` : `${gb.toFixed(1)} GB`;
}

function getErrorMessage(error: unknown, fallback: string) {
  return error instanceof Error && error.message ? error.message : fallback;
}

function getBatchAcceptedCount(batch: BatchResponse) {
  return typeof batch.total_videos === "number" ? batch.total_videos : 0;
}

export function UploadDropzone({ onUploaded }: UploadDropzoneProps) {
  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [inlineMessage, setInlineMessage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [batchResult, setBatchResult] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const toast = useToast();

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const maxUploadSizeGb = getKnownMaxUploadSizeGb(settingsQuery.data);
  const maxUploadSizeBytes = maxUploadSizeGb ? maxUploadSizeGb * BYTES_PER_GB : undefined;
  const helpText = maxUploadSizeGb
    ? `Accepted formats: ${ACCEPTED_FORMATS}. Max file size: ${formatUploadCap(maxUploadSizeGb)}.`
    : `Accepted formats: ${ACCEPTED_FORMATS}.`;

  const totalSize = useMemo(
    () => files.reduce((total, item) => total + item.file.size, 0),
    [files],
  );
  const isBatchMode = files.length > 1;
  const uploadLabel = isBatchMode ? "Upload as batch" : "Upload video";
  const hasCompletedUpload = files.some((item) => item.status === "success");

  const updateFile = useCallback((id: string, next: Partial<SelectedFile>) => {
    setFiles((current) =>
      current.map((item) => (item.id === id ? { ...item, ...next } : item)),
    );
  }, []);

  const updateAllFiles = useCallback((next: Partial<SelectedFile>) => {
    setFiles((current) => current.map((item) => ({ ...item, ...next })));
  }, []);

  const invalidateUploadQueries = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["videos"] }),
      queryClient.invalidateQueries({ queryKey: ["jobs"] }),
      queryClient.invalidateQueries({ queryKey: ["batches"] }),
    ]);
  }, [queryClient]);

  const closeAfterConfirmation = useCallback(() => {
    window.setTimeout(() => {
      setFiles([]);
      setInlineMessage(null);
      setBatchResult(null);
      onUploaded?.();
    }, CLOSE_DELAY_MS);
  }, [onUploaded]);

  const onDrop = useCallback(
    (acceptedFiles: File[], fileRejections: FileRejection[]) => {
      const messages: string[] = [];
      const acceptedWithinLimit = acceptedFiles.filter((file) => {
        if (maxUploadSizeBytes && file.size > maxUploadSizeBytes) {
          messages.push(`${file.name} is over the ${formatUploadCap(maxUploadSizeGb ?? 0)} upload limit.`);
          return false;
        }
        return true;
      });

      for (const rejection of fileRejections) {
        const invalidType = rejection.errors.some((error) => error.code === "file-invalid-type");
        const tooLarge = rejection.errors.some((error) => error.code === "file-too-large");
        if (tooLarge && maxUploadSizeGb) {
          messages.push(`${rejection.file.name} is over the ${formatUploadCap(maxUploadSizeGb)} upload limit.`);
        } else if (invalidType) {
          messages.push(`${rejection.file.name} is not supported. Use ${ACCEPTED_FORMATS}.`);
        } else {
          messages.push(`${rejection.file.name} could not be added.`);
        }
      }

      if (messages.length > 0) {
        setInlineMessage(messages.slice(0, 3).join(" "));
      } else {
        setInlineMessage(null);
      }

      if (acceptedWithinLimit.length === 0) return;

      setBatchResult(null);
      setFiles((current) => [
        ...current,
        ...acceptedWithinLimit.map((file) => ({
          file,
          id: createFileId(file),
          progress: 0,
          status: "ready" as const,
        })),
      ]);
    },
    [maxUploadSizeBytes, maxUploadSizeGb],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "video/*": ACCEPTED_EXTENSIONS,
    },
    disabled: isUploading,
    maxSize: maxUploadSizeBytes,
    multiple: true,
    onDrop,
  });

  const removeFile = (id: string) => {
    setFiles((current) => current.filter((item) => item.id !== id));
    setBatchResult(null);
  };

  const uploadSingle = async (item: SelectedFile) => {
    updateFile(item.id, {
      message: undefined,
      progress: 0,
      status: "uploading",
    });

    try {
      await api.uploadVideo(item.file, undefined, (event: UploadProgressEvent) => {
        updateFile(item.id, { progress: event.percent });
      });
      updateFile(item.id, {
        message: "Uploaded and queued for processing.",
        progress: 100,
        status: "success",
      });
      await invalidateUploadQueries();
      toast({
        title: "Upload complete",
        description: "1 video uploaded and queued for processing.",
        variant: "success",
      });
      closeAfterConfirmation();
    } catch (error) {
      const message = getErrorMessage(error, "Upload failed");
      updateFile(item.id, {
        message,
        progress: 0,
        status: "error",
      });
      toast({
        title: "Upload failed",
        description: message,
        variant: "error",
      });
      toast({
        title: "Upload summary",
        description: "0 of 1 videos uploaded. Keep the file here and retry after fixing the issue.",
        variant: "info",
      });
    }
  };

  const uploadBatch = async (items: SelectedFile[]) => {
    const batchName = `Batch upload ${new Date().toLocaleString()}`;
    updateAllFiles({
      message: undefined,
      progress: 0,
      status: "uploading",
    });

    try {
      const batch = await api.createBatch(
        items.map((item) => item.file),
        batchName,
        (event: UploadProgressEvent) => {
          updateAllFiles({ progress: event.percent });
        },
      );
      const acceptedCount = getBatchAcceptedCount(batch);
      const result = `${acceptedCount} video${acceptedCount === 1 ? "" : "s"} accepted in ${batch.name ?? batchName}.`;
      setBatchResult(result);
      updateAllFiles({
        message: "Accepted in batch.",
        progress: 100,
        status: "success",
      });
      await invalidateUploadQueries();
      toast({
        title: "Batch accepted",
        description: result,
        variant: "success",
      });
      closeAfterConfirmation();
    } catch (error) {
      const message = getErrorMessage(error, "Batch upload failed");
      updateAllFiles({
        message,
        status: "error",
      });
      toast({
        title: "Batch upload failed",
        description: message,
        variant: "error",
      });
    }
  };

  const uploadAll = async () => {
    if (files.length === 0 || isUploading) return;
    setIsUploading(true);
    setInlineMessage(null);
    setBatchResult(null);
    try {
      if (files.length === 1) {
        await uploadSingle(files[0]);
      } else {
        await uploadBatch(files);
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <section className="space-y-4">
      <div
        {...getRootProps({
          "aria-describedby": "upload-dropzone-help",
        })}
        className={cn(
          "flex min-h-64 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed bg-card p-8 text-center transition duration-200 ease-out",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          isDragActive ? "border-info bg-info/10" : "border-border hover:bg-muted",
          isUploading && "cursor-not-allowed opacity-70",
        )}
      >
        <input {...getInputProps()} />
        <div className="flex h-14 w-14 items-center justify-center rounded-full border border-border bg-muted">
          <Upload className={cn("h-7 w-7", isDragActive ? "text-info" : "text-muted-foreground")} aria-hidden="true" />
        </div>
        <div className="mt-4 text-lg font-semibold text-card-foreground">
          {isDragActive ? "Drop videos here" : "Drag and drop videos here"}
        </div>
        <p id="upload-dropzone-help" className="mt-2 max-w-md text-sm text-muted-foreground">
          Browse or drop files. {helpText}
        </p>
        <Button className="mt-5" disabled={isUploading} size="sm" variant="outline">
          <Upload className="h-4 w-4" aria-hidden="true" />
          Browse files
        </Button>
      </div>

      {inlineMessage ? (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/25 bg-destructive/10 px-3 py-2 text-sm text-foreground">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" aria-hidden="true" />
          <p>{inlineMessage}</p>
        </div>
      ) : null}

      {files.length > 0 ? (
        <div className="overflow-hidden rounded-lg border border-border bg-card shadow-sm">
          <div className="flex flex-col gap-2 border-b border-border px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-sm font-semibold text-card-foreground">
                {files.length} file{files.length === 1 ? "" : "s"} selected
              </h3>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {formatFileSize(totalSize)} total
                {isBatchMode ? ". Multiple files upload as one server-side batch." : "."}
              </p>
            </div>
            <Badge variant={isBatchMode ? "info" : "outline"}>
              {isBatchMode ? "Batch upload" : "Single upload"}
            </Badge>
          </div>

          {batchResult ? (
            <div className="flex items-start gap-2 border-b border-border bg-success/10 px-4 py-3 text-sm text-card-foreground">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" aria-hidden="true" />
              <p>{batchResult}</p>
            </div>
          ) : null}

          <ul className="divide-y divide-border">
            {files.map((item) => {
              const progress = Math.round(item.progress);
              const canRemove = !isUploading && item.status !== "uploading";
              return (
                <li key={item.id} className="px-4 py-3">
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-muted">
                      <FileVideo className="h-5 w-5 text-muted-foreground" aria-hidden="true" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-card-foreground">{item.file.name}</p>
                          <p className="mt-0.5 text-xs text-muted-foreground">{formatFileSize(item.file.size)}</p>
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          <FileStatusBadge status={item.status} />
                          {canRemove ? (
                            <Button
                              aria-label={`Remove ${item.file.name}`}
                              className="h-8 w-8 p-0"
                              onClick={() => removeFile(item.id)}
                              size="sm"
                              variant="ghost"
                            >
                              <Trash2 className="h-4 w-4" aria-hidden="true" />
                            </Button>
                          ) : null}
                        </div>
                      </div>

                      {item.status === "uploading" ? (
                        <div className="mt-3 space-y-1.5">
                          <Progress value={progress} />
                          <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span>{isBatchMode ? "Batch upload progress" : "Uploading"}</span>
                            <span>{progress}%</span>
                          </div>
                        </div>
                      ) : null}

                      {item.message ? (
                        <p
                          className={cn(
                            "mt-2 flex items-start gap-1.5 text-xs",
                            item.status === "error" ? "text-destructive" : "text-muted-foreground",
                          )}
                        >
                          {item.status === "error" ? (
                            <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                          ) : (
                            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                          )}
                          <span>{item.message}</span>
                        </p>
                      ) : null}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>

          <div className="flex flex-col gap-2 border-t border-border px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs text-muted-foreground">
              {isBatchMode
                ? "Batch uploads enqueue all selected videos server-side. Processing concurrency is controlled by the server."
                : "Single uploads are queued for processing after the file transfer completes."}
            </p>
            <Button
              className="w-full sm:w-auto"
              disabled={files.length === 0 || hasCompletedUpload}
              loading={isUploading}
              onClick={uploadAll}
            >
              {files.some((item) => item.status === "error") ? (
                <RotateCcw className="h-4 w-4" aria-hidden="true" />
              ) : (
                <Upload className="h-4 w-4" aria-hidden="true" />
              )}
              {files.some((item) => item.status === "error") ? "Retry upload" : uploadLabel}
            </Button>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function FileStatusBadge({ status }: { status: UploadStatus }) {
  if (status === "uploading") {
    return <Badge variant="info">Uploading</Badge>;
  }
  if (status === "success") {
    return <Badge variant="success">Accepted</Badge>;
  }
  if (status === "error") {
    return <Badge variant="destructive">Failed</Badge>;
  }
  return <Badge variant="muted">Ready</Badge>;
}
