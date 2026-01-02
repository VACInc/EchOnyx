"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, FileVideo, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export function UploadDropzone() {
  const [files, setFiles] = useState<File[]>([]);
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadVideo(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["videos"] });
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/*": [".mp4", ".webm", ".mov", ".avi", ".mkv"],
    },
    multiple: true,
  });

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const uploadAll = async () => {
    for (const file of files) {
      await uploadMutation.mutateAsync(file);
    }
    setFiles([]);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-10 transition",
          isDragActive
            ? "border-blue-500 bg-blue-50"
            : "border-slate-300 hover:border-blue-400 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800/40"
        )}
      >
        <input {...getInputProps()} />
        <Upload
          className={cn(
            "h-12 w-12",
            isDragActive ? "text-blue-500" : "text-gray-400"
          )}
        />
        <p className="mt-4 text-lg font-medium text-gray-700">
          {isDragActive ? "Drop videos here" : "Drag & drop videos here"}
        </p>
        <p className="mt-1 text-sm text-gray-500 dark:text-slate-400">
          or click to browse (MP4, WebM, MOV, AVI, MKV)
        </p>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="rounded-2xl border border-slate-200/70 bg-white/80 dark:border-slate-700/60 dark:bg-slate-900/70">
          <div className="border-b border-slate-200/70 px-4 py-3 dark:border-slate-700/60">
            <h3 className="font-medium text-slate-900 dark:text-slate-100">
              {files.length} file{files.length > 1 ? "s" : ""} selected
            </h3>
          </div>
          <ul className="divide-y divide-slate-200/70 dark:divide-slate-700/60">
            {files.map((file, index) => (
              <li key={index} className="flex items-center px-4 py-3">
                <FileVideo className="h-8 w-8 text-slate-400 dark:text-slate-500" />
                <div className="ml-3 flex-1 min-w-0">
                  <p className="truncate text-sm font-medium text-slate-900 dark:text-slate-100">
                    {file.name}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {formatFileSize(file.size)}
                  </p>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="ml-4 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800"
                >
                  <X className="h-5 w-5" />
                </button>
              </li>
            ))}
          </ul>
          <div className="border-t border-slate-200/70 px-4 py-3 dark:border-slate-700/60">
            <button
              onClick={uploadAll}
              disabled={uploadMutation.isPending}
              className="flex w-full items-center justify-center rounded-full bg-slate-900 px-4 py-2 font-medium text-white transition hover:bg-slate-800 disabled:opacity-50 dark:bg-white dark:text-slate-900"
            >
              {uploadMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-5 w-5" />
                  Upload & Process All
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
