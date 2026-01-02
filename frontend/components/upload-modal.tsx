"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { UploadDropzone } from "@/components/upload-dropzone";
import { X, Sparkles, Shield, Film } from "lucide-react";
import { cn } from "@/lib/utils";

type UploadModalContextValue = {
  isOpen: boolean;
  openModal: () => void;
  closeModal: () => void;
};

const UploadModalContext = createContext<UploadModalContextValue | null>(null);

export function useUploadModal() {
  const ctx = useContext(UploadModalContext);
  if (!ctx) {
    throw new Error("useUploadModal must be used within UploadModalProvider");
  }
  return ctx;
}

export function UploadModalProvider({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const openModal = useCallback(() => setIsOpen(true), []);
  const closeModal = useCallback(() => setIsOpen(false), []);
  const value = useMemo(
    () => ({
      isOpen,
      openModal,
      closeModal,
    }),
    [isOpen, openModal, closeModal]
  );

  return (
    <UploadModalContext.Provider value={value}>
      {children}
      <UploadModal isOpen={isOpen} onClose={closeModal} />
    </UploadModalContext.Provider>
  );
}

function UploadModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className={cn(
          "relative w-full max-w-5xl overflow-hidden rounded-2xl border border-stone-200/70 bg-white shadow-2xl",
          "dark:border-slate-700/60 dark:bg-slate-900"
        )}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-stone-200/70 px-6 py-4 dark:border-slate-700/60">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-stone-500 dark:text-slate-400">
              Upload
            </p>
            <h2 className="text-xl font-semibold text-stone-900 dark:text-slate-100">
              Add new videos
            </h2>
          </div>
          <button
            onClick={onClose}
            className="rounded-full border border-stone-200/70 p-2 text-stone-500 transition hover:text-stone-800 dark:border-slate-700/60 dark:text-slate-300 dark:hover:text-slate-100"
            aria-label="Close upload modal"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="grid gap-6 p-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
          <div className="space-y-4">
            <UploadDropzone />
          </div>

          <div className="space-y-4">
            <div className="rounded-xl border border-stone-200/70 bg-stone-50/80 p-4 dark:border-slate-700/60 dark:bg-slate-800/60">
              <div className="flex items-center gap-2 text-sm font-semibold text-stone-900 dark:text-slate-100">
                <Film className="h-4 w-4 text-stone-600 dark:text-slate-300" />
                Processing pipeline
              </div>
              <ul className="mt-3 space-y-2 text-sm text-stone-600 dark:text-slate-300">
                <li>1. Extract audio + key frames</li>
                <li>2. Transcribe and diarize speakers</li>
                <li>3. Analyze slides + visuals</li>
                <li>4. Summarize and index for search</li>
              </ul>
            </div>

            <div className="rounded-xl border border-amber-200/70 bg-amber-50/80 p-4 dark:border-amber-300/20 dark:bg-amber-500/10">
              <div className="flex items-center gap-2 text-sm font-semibold text-amber-900 dark:text-amber-200">
                <Sparkles className="h-4 w-4" />
                Tips
              </div>
              <ul className="mt-3 space-y-2 text-sm text-amber-900/80 dark:text-amber-100/80">
                <li>Use clear recordings for best transcription.</li>
                <li>Upload multiple videos; they will process sequentially.</li>
                <li>Tag videos after processing for faster retrieval.</li>
              </ul>
            </div>

            <div className="flex items-start gap-3 rounded-xl border border-stone-200/70 bg-white/70 p-4 text-xs text-stone-500 dark:border-slate-700/60 dark:bg-slate-800/40 dark:text-slate-400">
              <Shield className="mt-0.5 h-4 w-4 text-stone-400 dark:text-slate-400" />
              <p>
                Processing is fully local. No data leaves your machine unless you
                configured external endpoints.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
