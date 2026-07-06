"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { UploadModalProvider } from "@/components/upload-modal";
import { ConfirmDialogProvider } from "@/components/ui/confirm-dialog";
import { ToastProvider } from "@/components/ui/toast";

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <ConfirmDialogProvider>
          <UploadModalProvider>{children}</UploadModalProvider>
        </ConfirmDialogProvider>
      </ToastProvider>
    </QueryClientProvider>
  );
}
