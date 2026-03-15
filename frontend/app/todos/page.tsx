"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckSquare, Square, Trash2 } from "lucide-react";
import { api } from "@/lib/api";

export default function TodosPage() {
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<"open" | "completed" | "all">("open");
  const [sort, setSort] = useState<"updated_at" | "created_at" | "completed_at" | "video_title">("updated_at");
  const [search, setSearch] = useState("");
  const [tagInput, setTagInput] = useState("");
  const [tags, setTags] = useState<string[]>([]);

  const { data: settings } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const { data, isLoading } = useQuery({
    queryKey: ["action-items", { status, sort, search, tags }],
    queryFn: () => api.getActionItems({ status, sort, search, tags, pageSize: 100 }),
    enabled: settings?.action_items.enabled !== false,
  });

  const invalidate = () => Promise.all([
    queryClient.invalidateQueries({ queryKey: ["action-items"] }),
    queryClient.invalidateQueries({ queryKey: ["video"] }),
    queryClient.invalidateQueries({ queryKey: ["summary"] }),
  ]);

  const updateMutation = useMutation({
    mutationFn: ({ id, completed }: { id: string; completed: boolean }) => api.updateActionItem(id, { completed }),
    onSuccess: invalidate,
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteActionItem(id),
    onSuccess: invalidate,
  });

  const activeTags = useMemo(() => new Set(tags.map((tag) => tag.toLowerCase())), [tags]);

  const addTagsFromInput = () => {
    const next = tagInput
      .split(",")
      .map((tag) => tag.trim())
      .filter((tag) => tag.length > 0 && !activeTags.has(tag.toLowerCase()));
    if (!next.length) return;
    setTags((current) => [...current, ...next]);
    setTagInput("");
  };

  if (settings && !settings.action_items.enabled) {
    return <div className="py-12 text-center text-slate-500 dark:text-slate-400">Todos are disabled in Settings.</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Todos</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">Action items collected from videos and manual notes.</p>
        </div>
        <div className="text-sm text-slate-500 dark:text-slate-400">{data?.total ?? 0} items</div>
      </div>

      <div className="grid gap-3 rounded-2xl border border-slate-200/70 bg-white/80 p-4 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70 lg:grid-cols-[140px_180px_minmax(0,1fr)_minmax(0,1fr)]">
        <select
          value={status}
          onChange={(event) => setStatus(event.target.value as typeof status)}
          className="rounded-xl border border-slate-200 px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
        >
          <option value="open">Open</option>
          <option value="completed">Completed</option>
          <option value="all">All</option>
        </select>
        <select
          value={sort}
          onChange={(event) => setSort(event.target.value as typeof sort)}
          className="rounded-xl border border-slate-200 px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
        >
          <option value="updated_at">Recently updated</option>
          <option value="created_at">Newest</option>
          <option value="completed_at">Recently completed</option>
          <option value="video_title">Video title</option>
        </select>
        <input
          type="text"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Filter by todo text"
          className="rounded-xl border border-slate-200 px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
        />
        <div className="flex gap-2">
          <input
            type="text"
            value={tagInput}
            onChange={(event) => setTagInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addTagsFromInput();
              }
            }}
            placeholder="Filter by labels"
            className="flex-1 rounded-xl border border-slate-200 px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
          />
          <button
            type="button"
            onClick={addTagsFromInput}
            className="rounded-full border border-slate-200 px-4 py-2 text-sm dark:border-slate-700"
          >
            Add
          </button>
        </div>
      </div>

      {tags.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {tags.map((tag) => (
            <button
              key={tag}
              type="button"
              onClick={() => setTags((current) => current.filter((item) => item !== tag))}
              className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200"
            >
              {tag} ×
            </button>
          ))}
        </div>
      ) : null}

      {isLoading ? (
        <div className="py-12 text-center text-slate-500 dark:text-slate-400">Loading...</div>
      ) : data?.items.length ? (
        <div className="space-y-3">
          {data.items.map((item) => (
            <div
              key={item.id}
              className="flex items-start justify-between gap-4 rounded-2xl border border-slate-200/70 bg-white/80 p-4 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70"
            >
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => updateMutation.mutate({ id: item.id, completed: !item.completed })}
                  className="mt-0.5 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100"
                >
                  {item.completed ? <CheckSquare className="h-5 w-5 text-emerald-600" /> : <Square className="h-5 w-5" />}
                </button>
                <div className="space-y-2">
                  <p className={`text-sm ${item.completed ? "text-slate-400 line-through dark:text-slate-500" : "text-slate-800 dark:text-slate-100"}`}>
                    {item.text}
                  </p>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                    <Link href={`/videos/${item.video_id}`} className="font-medium text-blue-600 hover:underline">
                      {item.video_title}
                    </Link>
                    <span className="rounded-full bg-slate-100 px-2 py-1 dark:bg-slate-800">{item.source}</span>
                    {item.labels.map((label) => (
                      <span key={`${item.id}-${label}`} className="rounded-full bg-slate-100 px-2 py-1 dark:bg-slate-800">
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <button
                type="button"
                onClick={() => deleteMutation.mutate(item.id)}
                className="rounded-full border border-red-200 p-2 text-red-600 hover:bg-red-50 dark:border-red-500/30 dark:text-red-300"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="py-12 text-center text-slate-500 dark:text-slate-400">No todos match the current filters.</div>
      )}
    </div>
  );
}
