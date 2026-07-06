"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import { Film, Lightbulb, ShieldCheck, Sparkles } from "lucide-react";

import { UploadDropzone } from "@/components/upload-dropzone";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";

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
    [isOpen, openModal, closeModal],
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
  return (
    <Dialog
      className="max-w-5xl"
      description="Upload one video directly, or send multiple videos as a server-side batch."
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      open={isOpen}
      title="Add videos"
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(18rem,0.75fr)]">
        <UploadDropzone onUploaded={onClose} />

        <aside className="space-y-4">
          <Card className="p-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-card-foreground">
              <Film className="h-4 w-4 text-info" aria-hidden="true" />
              Processing pipeline
            </div>
            <ol className="mt-3 space-y-2 text-sm text-muted-foreground">
              <li>1. Extract audio and key frames.</li>
              <li>2. Transcribe speech and identify speakers.</li>
              <li>3. Analyze slides and visual changes.</li>
              <li>4. Summarize, index, and queue search data.</li>
            </ol>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-card-foreground">
                <Lightbulb className="h-4 w-4 text-warning" aria-hidden="true" />
                Tips
              </div>
              <Badge variant="info">Batch aware</Badge>
            </div>
            <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
              <li>Clear audio improves transcription and speaker labels.</li>
              <li>Multiple files are accepted as one batch and enqueued server-side.</li>
              <li>Batch concurrency follows the server setting for concurrent jobs.</li>
              <li>Add tags after processing to make recurring searches easier.</li>
            </ul>
          </Card>

          <Card className="p-4">
            <div className="flex items-start gap-3">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-success" aria-hidden="true" />
              <div>
                <div className="text-sm font-semibold text-card-foreground">Privacy</div>
                <p className="mt-1 text-sm text-muted-foreground">
                  Processing runs locally. Data only leaves this machine if external
                  endpoints are configured.
                </p>
              </div>
            </div>
          </Card>

          <div className="flex items-center gap-2 rounded-lg border border-border bg-muted px-3 py-2 text-xs text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5 text-info" aria-hidden="true" />
            New uploads appear in videos and jobs after the server accepts them.
          </div>
        </aside>
      </div>
    </Dialog>
  );
}
