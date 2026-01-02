"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  Upload,
  Video,
  Search,
  Settings,
  Cpu,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useUploadModal } from "@/components/upload-modal";
import { ThemeToggle } from "@/components/theme-toggle";

const navigation = [
  { name: "Dashboard", href: "/", icon: Home },
  { name: "Videos", href: "/videos", icon: Video },
  { name: "Search", href: "/search", icon: Search },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { openModal } = useUploadModal();

  return (
    <div className="flex h-full w-64 flex-col bg-slate-950">
      {/* Logo */}
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center">
          <Cpu className="h-8 w-8 text-blue-400" />
          <span className="ml-3 text-xl font-semibold text-white">
            Video Summarizer
          </span>
        </div>
      </div>

      <div className="px-6">
        <button
          onClick={openModal}
          className="flex w-full items-center justify-center rounded-full bg-blue-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-400"
        >
          <Upload className="mr-2 h-4 w-4" />
          Upload Video
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "group flex items-center rounded-lg px-3 py-2 text-sm font-medium transition",
                isActive
                  ? "bg-slate-800 text-white"
                  : "text-slate-300 hover:bg-slate-900 hover:text-white"
              )}
            >
              <item.icon
                className={cn(
                  "mr-3 h-5 w-5",
                  isActive ? "text-blue-300" : "text-slate-500 group-hover:text-white"
                )}
              />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-slate-800 p-4 space-y-3">
        <ThemeToggle />
        <div>
          <p className="text-xs text-slate-400">Fully Local Processing</p>
          <p className="text-xs text-slate-500">No data leaves your machine</p>
        </div>
      </div>
    </div>
  );
}
