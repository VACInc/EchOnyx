"use client";

import { useMemo, useState } from "react";
import { Search, Upload, Video } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { VideoCard } from "@/components/video-card";
import { useUploadModal } from "@/components/upload-modal";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";

type Job = Awaited<ReturnType<typeof api.getJobs>>["jobs"][number];

const PAGE_SIZE = 20;

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function VideoListSkeleton() {
  return (
    <div className="space-y-3">
      {Array.from({ length: 6 }).map((_, index) => (
        <div key={index} className="rounded-lg border border-border bg-card p-4 shadow-sm">
          <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_150px_120px_120px_140px] md:items-center">
            <div className="flex items-start gap-3">
              <Skeleton className="h-10 w-10 shrink-0 rounded-lg" />
              <div className="min-w-0 flex-1 space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-5 w-36 rounded-full" />
              </div>
            </div>
            <Skeleton className="h-6 w-28 rounded-full" />
            <Skeleton className="h-4 w-20" />
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-24" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function VideosPage() {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const { openModal } = useUploadModal();

  const videosQuery = useQuery({
    queryKey: ["videos", page, search],
    queryFn: () => api.getVideos({ page, pageSize: PAGE_SIZE, search: search || undefined }),
  });

  const processingJobsQuery = useQuery({
    queryKey: ["jobs", "processing"],
    queryFn: () => api.getJobs({ status: "processing", pageSize: 100 }),
    refetchInterval: 2000,
  });

  const queuedJobsQuery = useQuery({
    queryKey: ["jobs", "queued"],
    queryFn: () => api.getJobs({ status: "queued", pageSize: 100 }),
    refetchInterval: 2000,
  });

  const jobByVideoId = useMemo(() => {
    const map: Record<string, Job> = {};
    for (const job of [...(processingJobsQuery.data?.jobs ?? []), ...(queuedJobsQuery.data?.jobs ?? [])]) {
      map[job.video_id] = job;
    }
    return map;
  }, [processingJobsQuery.data?.jobs, queuedJobsQuery.data?.jobs]);

  const totalPages = videosQuery.data ? Math.ceil(videosQuery.data.total / PAGE_SIZE) : 1;
  const hasVideos = (videosQuery.data?.videos?.length ?? 0) > 0;

  return (
    <div className="space-y-6">
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
        <Input
          type="text"
          placeholder="Search videos"
          value={search}
          onChange={(event) => {
            setSearch(event.target.value);
            setPage(1);
          }}
          className="pl-9"
          aria-label="Search videos"
        />
      </div>

      <Card className="p-4">
        <div className="hidden border-b border-border pb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground md:grid md:grid-cols-[minmax(0,2fr)_150px_120px_120px_140px] md:gap-4">
          <span>Video</span>
          <span>Status</span>
          <span>Duration</span>
          <span>Size</span>
          <span>Added</span>
        </div>

        <div className="mt-4">
          {videosQuery.isLoading ? (
            <VideoListSkeleton />
          ) : videosQuery.isError ? (
            <ErrorState
              title="Videos failed to load"
              message={getErrorMessage(videosQuery.error, "Refresh to try loading videos again.")}
              onRetry={() => videosQuery.refetch()}
            />
          ) : !hasVideos ? (
            search ? (
              <EmptyState
                icon={<Search className="h-6 w-6" aria-hidden="true" />}
                headline="No matches"
                hint="Try a different title, filename, tag, or transcript term."
              />
            ) : (
              <EmptyState
                icon={<Video className="h-6 w-6" aria-hidden="true" />}
                headline="No videos yet"
                hint="Upload a video to start building your searchable local library."
                action={
                  <Button onClick={openModal}>
                    <Upload className="h-4 w-4" aria-hidden="true" />
                    Upload
                  </Button>
                }
              />
            )
          ) : (
            <div className="space-y-3">
              {videosQuery.data?.videos?.map((video) => (
                <VideoCard key={video.id} video={video} job={jobByVideoId[video.id]} />
              ))}
            </div>
          )}
        </div>
      </Card>

      {videosQuery.data && videosQuery.data.total > PAGE_SIZE ? (
        <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Button
            onClick={() => setPage((currentPage) => Math.max(1, currentPage - 1))}
            disabled={page === 1}
            variant="outline"
          >
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {page} of {totalPages}
          </span>
          <Button
            onClick={() => setPage((currentPage) => currentPage + 1)}
            disabled={page >= totalPages}
            variant="outline"
          >
            Next
          </Button>
        </div>
      ) : null}
    </div>
  );
}
