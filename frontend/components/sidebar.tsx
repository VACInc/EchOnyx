"use client";

import { useQuery } from "@tanstack/react-query";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  ListTodo,
  LogOut,
  Search,
  Settings,
  Upload,
  Video,
} from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/components/auth-gate";
import { cn } from "@/lib/utils";
import { useUploadModal } from "@/components/upload-modal";
import { ThemeToggle } from "@/components/theme-toggle";

const navigation = [
  { name: "Dashboard", href: "/", icon: Home },
  { name: "Videos", href: "/videos", icon: Video },
  { name: "Search", href: "/search", icon: Search },
  { name: "Todos", href: "/todos", icon: ListTodo, requiresActionItems: true },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { openModal } = useUploadModal();
  const { logout } = useAuth();
  const { data: settings } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });
  const actionItemsEnabled = settings?.action_items.enabled ?? true;

  return (
    <div className="flex h-full w-64 flex-col bg-gradient-to-b from-[#0a0d18] via-[#0e1326] to-[#0b0f1a]">
      {/* Logo */}
      <div className="flex h-20 items-center justify-between px-4">
        <div className="flex h-full w-full items-center">
          <Image
            src="/echonyx-horizontal.png"
            alt="EchOnyx"
            width={240}
            height={80}
            className="h-full w-full object-contain"
          />
        </div>
      </div>

      <div className="px-6">
        <button
          onClick={openModal}
          className="flex w-full items-center justify-center rounded-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-blue-500/20 transition hover:from-blue-400 hover:via-indigo-400 hover:to-purple-400"
        >
          <Upload className="mr-2 h-4 w-4" />
          Upload Video
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.filter((item) => !item.requiresActionItems || actionItemsEnabled).map((item) => {
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
                  ? "bg-slate-800/80 text-white shadow-inner shadow-blue-500/10"
                  : "text-slate-300 hover:bg-slate-900/80 hover:text-white"
              )}
            >
              <item.icon
                className={cn(
                  "mr-3 h-5 w-5",
                  isActive ? "text-blue-300" : "text-slate-500 group-hover:text-blue-200"
                )}
              />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-slate-800/80 p-4 space-y-3">
        <ThemeToggle />
        <button
          type="button"
          onClick={() => void logout()}
          className="flex w-full items-center justify-center rounded-full border border-slate-700 px-3 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-300 transition hover:border-slate-500 hover:text-white"
        >
          <LogOut className="mr-2 h-4 w-4" />
          Sign out
        </button>
        <div>
          <p className="text-xs text-slate-400">Fully Local Processing</p>
          <p className="text-xs text-slate-500">No data leaves your machine</p>
        </div>
      </div>
    </div>
  );
}
