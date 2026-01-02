import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Sidebar } from "@/components/sidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Video Summarizer",
  description: "Local video and presentation summarization system",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Providers>
          <div className="flex h-screen">
            <Sidebar />
            <main className="flex-1 overflow-auto bg-gradient-to-b from-stone-50 via-amber-50/30 to-stone-100 p-6 dark:from-slate-950 dark:via-slate-900/40 dark:to-slate-950">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
