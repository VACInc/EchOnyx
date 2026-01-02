import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Sidebar } from "@/components/sidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "EchOnyx",
  description: "Privacy-first video and presentation intelligence",
  icons: {
    icon: "/favicon.ico",
    apple: "/apple-touch-icon.png",
  },
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
            <main className="flex-1 overflow-auto bg-gradient-to-b from-slate-50 via-blue-50/40 to-indigo-50 p-6 dark:from-[#070a14] dark:via-[#0f152b]/70 dark:to-[#0b0f1a]">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
