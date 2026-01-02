"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { VideoCard } from "@/components/video-card";
import { Search, Video } from "lucide-react";

export default function VideosPage() {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  const { data, isLoading } = useQuery({
    queryKey: ["videos", page, search],
    queryFn: () => api.getVideos({ page, pageSize: 20, search: search || undefined }),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold text-stone-900 dark:text-slate-100">Videos</h1>
          <p className="mt-1 text-sm text-stone-500 dark:text-slate-400">
            {data?.total ?? 0} videos in library
          </p>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-gray-400" />
        <input
          type="text"
          placeholder="Search videos..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
          className="w-full rounded-2xl border border-stone-200 bg-white/80 py-3 pl-10 pr-4 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-900/70"
        />
      </div>

      {/* Video List */}
      {isLoading ? (
        <div className="py-12 text-center text-stone-500 dark:text-slate-400">Loading...</div>
      ) : data?.videos?.length === 0 ? (
        <div className="py-12 text-center">
          <Video className="mx-auto h-12 w-12 text-stone-400 dark:text-slate-500" />
          <p className="mt-2 text-stone-500 dark:text-slate-400">
            {search ? "No videos match your search" : "No videos yet"}
          </p>
        </div>
      ) : (
        <div className="rounded-2xl border border-stone-200/70 bg-white/70 p-4 shadow-sm dark:border-slate-700/60 dark:bg-slate-900/60">
          <div className="hidden md:grid md:grid-cols-[minmax(0,2fr)_150px_120px_120px_140px] md:gap-4 border-b border-stone-200/70 pb-2 text-xs uppercase tracking-[0.2em] text-stone-400 dark:border-slate-700/60 dark:text-slate-500">
            <span>Video</span>
            <span>Status</span>
            <span>Duration</span>
            <span>Size</span>
            <span>Added</span>
          </div>
          <div className="mt-4 space-y-3">
            {data?.videos?.map((video) => (
              <VideoCard key={video.id} video={video} />
            ))}
          </div>
        </div>
      )}

      {/* Pagination */}
      {data && data.total > 20 && (
        <div className="flex items-center justify-center space-x-4">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="rounded-full border border-stone-200 px-4 py-2 text-sm font-medium text-stone-600 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300"
          >
            Previous
          </button>
          <span className="text-sm text-stone-600 dark:text-slate-300">
            Page {page} of {Math.ceil(data.total / 20)}
          </span>
          <button
            onClick={() => setPage((p) => p + 1)}
            disabled={page >= Math.ceil(data.total / 20)}
            className="rounded-full bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-white dark:text-slate-900"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
