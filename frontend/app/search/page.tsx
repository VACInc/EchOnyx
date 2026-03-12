"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Search, MessageSquare, Video } from "lucide-react";
import Link from "next/link";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"search" | "ask">("search");
  const [tagInput, setTagInput] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const warmedRef = useRef({ search: false, ask: false });

  const searchMutation = useMutation({
    mutationFn: ({ q, filterTags }: { q: string; filterTags: string[] }) =>
      api.search(q, undefined, filterTags),
  });

  const askMutation = useMutation({
    mutationFn: ({ q, filterTags }: { q: string; filterTags: string[] }) =>
      api.askQuestion(q, undefined, filterTags),
  });

  useEffect(() => {
    if (warmedRef.current.search) {
      return;
    }
    warmedRef.current.search = true;
    void api.warmSearchRuntime("search").catch(() => {
      warmedRef.current.search = false;
    });
  }, []);

  useEffect(() => {
    if (mode !== "ask") {
      return;
    }

    let cancelled = false;

    const warmAsk = async () => {
      if (cancelled || document.visibilityState !== "visible") {
        return;
      }
      try {
        await api.warmSearchRuntime("ask");
        warmedRef.current.ask = true;
      } catch {
        if (!cancelled) {
          warmedRef.current.ask = false;
        }
      }
    };

    void warmAsk();
    const intervalId = window.setInterval(() => {
      void warmAsk();
    }, 20_000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [mode]);

  const handleModeChange = (nextMode: "search" | "ask") => {
    setMode(nextMode);
    if (nextMode === "search" && warmedRef.current.search) {
      return;
    }
    if (nextMode === "ask" && warmedRef.current.ask) {
      return;
    }
    void api.warmSearchRuntime(nextMode).then(() => {
      warmedRef.current[nextMode] = true;
    }).catch(() => {
      warmedRef.current[nextMode] = false;
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    if (mode === "search") {
      searchMutation.mutate({ q: query, filterTags: tags });
    } else {
      askMutation.mutate({ q: query, filterTags: tags });
    }
  };

  const addTagsFromInput = () => {
    const candidates = tagInput
      .split(",")
      .map((tag) => tag.trim())
      .filter((tag) => tag.length > 0);
    if (candidates.length === 0) {
      return;
    }
    const existingLower = new Set(tags.map((tag) => tag.toLowerCase()));
    const merged = [...tags];
    for (const tag of candidates) {
      if (!existingLower.has(tag.toLowerCase())) {
        merged.push(tag);
        existingLower.add(tag.toLowerCase());
      }
    }
    setTags(merged);
    setTagInput("");
  };

  const removeTag = (tagToRemove: string) => {
    setTags(tags.filter((tag) => tag !== tagToRemove));
  };

  return (
    <div className="space-y-6">
      {/* Mode Toggle */}
      <div className="flex space-x-2">
        <button
          onClick={() => handleModeChange("search")}
          className={`flex items-center rounded-lg px-4 py-2 ${
            mode === "search"
              ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
              : "bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200"
          }`}
        >
          <Search className="mr-2 h-4 w-4" />
          Search
        </button>
        <button
          onClick={() => handleModeChange("ask")}
          className={`flex items-center rounded-lg px-4 py-2 ${
            mode === "ask"
              ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
              : "bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200"
          }`}
        >
          <MessageSquare className="mr-2 h-4 w-4" />
          Ask Question
        </button>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder={
                mode === "search"
                  ? "Search transcripts..."
                  : "Ask a question about your videos..."
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full rounded-2xl border border-slate-200 bg-white/80 py-3 pl-10 pr-4 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-900/70"
            />
          </div>
          <button
            type="submit"
            disabled={!query.trim() || searchMutation.isPending || askMutation.isPending}
            className="rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50 dark:bg-white dark:text-slate-900"
          >
            {mode === "search" ? "Search" : "Ask"}
          </button>
        </div>
        <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-4 dark:border-slate-700/60 dark:bg-slate-900/70">
          <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Filter by labels</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {tags.length === 0 && (
              <span className="text-sm text-slate-400 dark:text-slate-500">No label filters applied.</span>
            )}
            {tags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200"
              >
                {tag}
                <button
                  type="button"
                  onClick={() => removeTag(tag)}
                  className="ml-2 text-slate-400 hover:text-slate-600 dark:text-slate-400 dark:hover:text-slate-200"
                  aria-label={`Remove ${tag}`}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
          <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center">
            <input
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  addTagsFromInput();
                }
              }}
              placeholder="Add labels to filter (comma-separated)"
              className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-900/70"
            />
            <button
              type="button"
              onClick={addTagsFromInput}
              disabled={tagInput.trim().length === 0}
              className="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
            >
              Add filter
            </button>
          </div>
        </div>
      </form>

      {/* Results */}
      {mode === "search" && searchMutation.data && (
        <div className="space-y-4">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {searchMutation.data.total} results for &quot;{searchMutation.data.query}&quot;
          </p>
          {tags.length > 0 && (
            <p className="text-xs text-slate-400 dark:text-slate-500">
              Filtered by labels: {tags.join(", ")}
            </p>
          )}
          {searchMutation.data.results.map((result, idx) => (
            <div
              key={idx}
              className="rounded-2xl border border-slate-200/70 bg-white/80 p-4 dark:border-slate-700/60 dark:bg-slate-900/70"
            >
              <div className="flex items-center justify-between">
                <Link
                  href={`/videos/${result.video_id}`}
                  className="flex items-center text-blue-600 hover:underline"
                >
                  <Video className="mr-2 h-4 w-4" />
                  {result.video_title}
                </Link>
                {result.timestamp_formatted && (
                  <span className="text-sm text-slate-500 dark:text-slate-400">
                    {result.timestamp_formatted}
                  </span>
                )}
              </div>
              {result.speaker && (
                <p className="mt-1 text-sm font-medium text-slate-700 dark:text-slate-200">
                  {result.speaker}
                </p>
              )}
              <p className="mt-2 text-slate-600 dark:text-slate-300">{result.text}</p>
              {result.context && (
                <p className="mt-1 text-sm text-slate-400 dark:text-slate-500">...{result.context}...</p>
              )}
            </div>
          ))}
        </div>
      )}

      {mode === "ask" && askMutation.data && (
        <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-6 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/70">
          <h3 className="font-medium text-slate-900 dark:text-slate-100">Answer</h3>
          <p className="mt-2 text-slate-700 dark:text-slate-200">{askMutation.data.answer}</p>

          {askMutation.data.sources.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-slate-500 dark:text-slate-400">Sources</h4>
              <ul className="mt-2 space-y-2">
                {askMutation.data.sources.map((source, idx) => (
                  <li key={idx} className="rounded-xl bg-slate-50 p-2 text-sm dark:bg-slate-800/70">
                    <Link
                      href={`/videos/${source.video_id}`}
                      className="text-blue-600 hover:underline"
                    >
                      {source.video_title}
                    </Link>
                    {source.timestamp_formatted && (
                      <span className="ml-2 text-gray-500">
                        @ {source.timestamp_formatted}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
