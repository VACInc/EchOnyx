"use client";

import type { ReactNode } from "react";
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
import { Button } from "@/components/ui/button";

export const navigation = [
  { name: "Dashboard", href: "/", icon: Home },
  { name: "Videos", href: "/videos", icon: Video },
  { name: "Search", href: "/search", icon: Search },
  { name: "Todos", href: "/todos", icon: ListTodo, requiresActionItems: true },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar({
  className,
  headerAction,
  onNavigate,
}: {
  className?: string;
  headerAction?: ReactNode;
  onNavigate?: () => void;
}) {
  const pathname = usePathname();
  const { openModal } = useUploadModal();
  const { logout } = useAuth();
  const { data: settings } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });
  const actionItemsEnabled = settings?.action_items.enabled ?? true;

  return (
    <div
      className={cn(
        "flex h-full w-64 shrink-0 flex-col bg-gradient-to-b from-[#0a0d18] via-[#0e1326] to-[#0b0f1a]",
        className,
      )}
    >
      {/* Logo */}
      <div className="flex h-20 items-center justify-between gap-2 px-4">
        <div className="flex h-full min-w-0 flex-1 items-center">
          <Image
            src="/echonyx-horizontal.png"
            alt="EchOnyx"
            width={240}
            height={80}
            className="h-full w-full object-contain"
          />
        </div>
        {headerAction}
      </div>

      <div className="px-6">
        <Button
          onClick={() => {
            openModal();
            onNavigate?.();
          }}
          className="w-full rounded-full shadow-lg shadow-blue-500/20"
        >
          <Upload className="mr-2 h-4 w-4" />
          Upload Video
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-4" aria-label="Primary navigation">
        {navigation.filter((item) => !item.requiresActionItems || actionItemsEnabled).map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href));

          return (
            <Link
              key={item.name}
              href={item.href}
              aria-current={isActive ? "page" : undefined}
              onClick={onNavigate}
              className={cn(
                "group flex items-center rounded-lg px-3 py-2 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-[#0b0f1a]",
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
      <div className="space-y-3 border-t border-slate-800/80 p-4">
        <ThemeToggle />
        <Button
          type="button"
          onClick={() => void logout()}
          variant="outline"
          size="sm"
          className="w-full rounded-full border-slate-700 bg-transparent text-slate-300 hover:border-slate-500 hover:bg-white/5 hover:text-white focus-visible:ring-offset-[#0b0f1a]"
        >
          <LogOut className="mr-2 h-4 w-4" />
          Sign out
        </Button>
        <div>
          <p className="text-xs text-slate-400">Fully Local Processing</p>
          <p className="text-xs text-slate-500">No data leaves your machine</p>
        </div>
      </div>
    </div>
  );
}
