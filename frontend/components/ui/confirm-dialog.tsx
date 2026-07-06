"use client";

import * as React from "react";

import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";

export interface ConfirmOptions {
  cancelLabel?: string;
  confirmLabel?: string;
  description?: React.ReactNode;
  destructive?: boolean;
  title: React.ReactNode;
}

type PendingConfirm = {
  options: ConfirmOptions;
  resolve: (confirmed: boolean) => void;
};

const ConfirmDialogContext = React.createContext<((options: ConfirmOptions) => Promise<boolean>) | null>(null);

export function ConfirmDialogProvider({ children }: { children: React.ReactNode }) {
  const [pendingConfirm, setPendingConfirm] = React.useState<PendingConfirm | null>(null);

  const confirm = React.useCallback((options: ConfirmOptions) => {
    return new Promise<boolean>((resolve) => {
      setPendingConfirm({ options, resolve });
    });
  }, []);

  const settle = React.useCallback(
    (confirmed: boolean) => {
      pendingConfirm?.resolve(confirmed);
      setPendingConfirm(null);
    },
    [pendingConfirm],
  );

  const options = pendingConfirm?.options;

  return (
    <ConfirmDialogContext.Provider value={confirm}>
      {children}
      <Dialog
        open={!!pendingConfirm}
        onOpenChange={(open) => {
          if (!open) settle(false);
        }}
        title={options?.title ?? "Confirm action"}
        description={options?.description}
        footer={
          <>
            <Button type="button" variant="outline" onClick={() => settle(false)}>
              {options?.cancelLabel ?? "Cancel"}
            </Button>
            <Button
              type="button"
              variant={options?.destructive ? "destructive" : "primary"}
              onClick={() => settle(true)}
            >
              {options?.confirmLabel ?? "Confirm"}
            </Button>
          </>
        }
      >
        <p className="text-sm text-muted-foreground">Choose whether to continue.</p>
      </Dialog>
    </ConfirmDialogContext.Provider>
  );
}

export function useConfirm() {
  const context = React.useContext(ConfirmDialogContext);
  if (!context) {
    throw new Error("useConfirm must be used within ConfirmDialogProvider");
  }
  return context;
}
