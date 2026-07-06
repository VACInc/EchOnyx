import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { AuthGate } from "@/components/auth-gate";
import { Sidebar } from "@/components/sidebar";

const inter = Inter({ subsets: ["latin"] });

const themeScript = `
(() => {
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  try {
    const stored = window.localStorage.getItem("echonyx-theme");
    const theme = stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
    document.documentElement.classList.toggle("dark", theme === "dark" || (theme === "system" && systemDark));
  } catch {
    document.documentElement.classList.toggle("dark", systemDark);
  }
})();
`;

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
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className={inter.className}>
        <Providers>
          <AuthGate>
            <div className="flex h-screen">
              <Sidebar />
              <main className="flex-1 overflow-auto bg-gradient-to-b from-slate-50 via-blue-50/40 to-indigo-50 p-6 dark:from-[#070a14] dark:via-[#0f152b]/70 dark:to-[#0b0f1a]">
                {children}
              </main>
            </div>
          </AuthGate>
        </Providers>
      </body>
    </html>
  );
}
