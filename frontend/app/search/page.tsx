"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  MessageSquare,
  MessageSquareText,
  RotateCcw,
  Search,
  SearchX,
  Tags,
  Video,
} from "lucide-react";

import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { TagInput } from "@/components/ui/tag-input";

type AskResponse = Awaited<ReturnType<typeof api.askQuestion>>;

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Array<{
    video_id: string;
    video_title: string;
    timestamp_formatted: string | null;
  }>;
};

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function SearchResultSkeleton() {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between gap-4">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <Skeleton className="h-4 w-4 shrink-0 rounded-full" />
          <Skeleton className="h-4 w-2/3" />
        </div>
        <Skeleton className="h-6 w-16 rounded-full" />
      </div>
      <Skeleton className="mt-3 h-4 w-24" />
      <div className="mt-3 space-y-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-11/12" />
        <Skeleton className="h-4 w-3/4" />
      </div>
    </Card>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"search" | "ask">("search");
  const [tags, setTags] = useState<string[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const warmedRef = useRef({ search: false, ask: false });

  const labelsQuery = useQuery({
    queryKey: ["video-labels"],
    queryFn: api.getVideoLabels,
  });

  const labelSuggestions = useMemo(
    () => labelsQuery.data?.labels.map((label) => label.name) ?? [],
    [labelsQuery.data],
  );

  const searchMutation = useMutation({
    mutationFn: ({ q, filterTags }: { q: string; filterTags: string[] }) =>
      api.search(q, undefined, filterTags),
  });

  const askMutation = useMutation({
    mutationFn: ({
      q,
      filterTags,
      history,
    }: {
      q: string;
      filterTags: string[];
      history: Array<{ role: "user" | "assistant"; content: string }>;
    }) => api.askQuestion(q, undefined, filterTags, history),
    onSuccess: (data: AskResponse, variables) => {
      setChatMessages((current) => [
        ...current,
        { role: "user", content: variables.q },
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources.map((source) => ({
            video_id: source.video_id,
            video_title: source.video_title,
            timestamp_formatted: source.timestamp_formatted,
          })),
        },
      ]);
      setQuery("");
    },
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
    void api
      .warmSearchRuntime(nextMode)
      .then(() => {
        warmedRef.current[nextMode] = true;
      })
      .catch(() => {
        warmedRef.current[nextMode] = false;
      });
  };

  const submitAsk = (q: string, filterTags: string[]) => {
    const trimmed = q.trim();
    const history = chatMessages.map(({ role, content }) => ({ role, content }));
    askMutation.mutate({ q: trimmed, filterTags, history });
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!query.trim()) return;

    if (mode === "search") {
      searchMutation.mutate({ q: query, filterTags: tags });
      return;
    }

    submitAsk(query, tags);
  };

  const retrySearch = () => {
    const variables = searchMutation.variables;
    if (variables) {
      searchMutation.mutate(variables);
    }
  };

  const retryAsk = () => {
    const variables = askMutation.variables;
    if (variables) {
      askMutation.mutate(variables);
    }
  };

  const searchData = searchMutation.data;
  const hasCompletedEmptySearch =
    mode === "search" &&
    searchMutation.isSuccess &&
    searchData !== undefined &&
    searchData.results.length === 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant={mode === "search" ? "primary" : "secondary"}
          onClick={() => handleModeChange("search")}
          aria-pressed={mode === "search"}
        >
          <Search className="h-4 w-4" aria-hidden="true" />
          Search
        </Button>
        <Button
          type="button"
          variant={mode === "ask" ? "primary" : "secondary"}
          onClick={() => handleModeChange("ask")}
          aria-pressed={mode === "ask"}
        >
          <MessageSquare className="h-4 w-4" aria-hidden="true" />
          Ask
        </Button>
        {mode === "ask" && chatMessages.length > 0 ? (
          <Button type="button" variant="outline" onClick={() => setChatMessages([])}>
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            New chat
          </Button>
        ) : null}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex flex-col gap-3 lg:flex-row">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              type="text"
              placeholder={
                mode === "search"
                  ? "Search transcripts..."
                  : chatMessages.length > 0
                    ? "Ask a follow-up about your videos..."
                    : "Ask a question about your videos..."
              }
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="pl-9"
            />
          </div>
          <Button
            type="submit"
            disabled={!query.trim() || searchMutation.isPending || askMutation.isPending}
            loading={mode === "search" ? searchMutation.isPending : askMutation.isPending}
            className="lg:min-w-28"
          >
            {mode === "search" ? "Search" : "Ask"}
          </Button>
        </div>

        <Card className="p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start">
            <div className="lg:w-56">
              <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                <Tags className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                Filter by labels
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                Suggestions come from labels already attached to videos.
              </p>
            </div>
            <div className="min-w-0 flex-1">
              <TagInput
                value={tags}
                onChange={setTags}
                suggestions={labelSuggestions}
                placeholder={labelsQuery.isLoading ? "Loading labels..." : "Add label filters"}
              />
            </div>
          </div>
        </Card>
      </form>

      {mode === "search" && searchMutation.isPending ? (
        <div className="space-y-4" aria-busy="true" aria-label="Loading search results">
          <SearchResultSkeleton />
          <SearchResultSkeleton />
          <SearchResultSkeleton />
        </div>
      ) : null}

      {mode === "search" && searchMutation.isError ? (
        <ErrorState
          title="Search failed"
          message={getErrorMessage(searchMutation.error, "Search could not be completed.")}
          onRetry={searchMutation.variables ? retrySearch : undefined}
          retryLabel="Retry search"
        />
      ) : null}

      {hasCompletedEmptySearch ? (
        <EmptyState
          icon={<SearchX className="h-6 w-6" aria-hidden="true" />}
          headline="No results found"
          hint={
            tags.length > 0
              ? "The current label filters may be narrowing the search too much."
              : "Try a different phrase or add more processed videos."
          }
        />
      ) : null}

      {mode === "search" && searchData && !searchMutation.isPending && searchData.results.length > 0 ? (
        <div className="space-y-4">
          <div className="flex flex-col gap-1 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
            <p>
              {searchData.total} results for &quot;{searchData.query}&quot;
            </p>
            {tags.length > 0 ? <p>Filtered by labels: {tags.join(", ")}</p> : null}
          </div>
          {searchData.results.map((result, index) => (
            <Card key={`${result.video_id}-${result.timestamp ?? index}-${index}`} className="p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <Link
                  href={`/videos/${result.video_id}`}
                  className="inline-flex min-w-0 items-center gap-2 text-sm font-semibold text-primary hover:underline"
                >
                  <Video className="h-4 w-4 shrink-0" aria-hidden="true" />
                  <span className="truncate">{result.video_title}</span>
                </Link>
                {result.timestamp_formatted ? (
                  <Badge variant="muted" className="w-fit shrink-0">
                    {result.timestamp_formatted}
                  </Badge>
                ) : null}
              </div>
              {result.speaker ? (
                <p className="mt-3 text-sm font-medium text-foreground">{result.speaker}</p>
              ) : null}
              <p className="mt-2 text-sm leading-6 text-card-foreground">{result.text}</p>
              {result.context ? (
                <p className="mt-2 border-l-2 border-border pl-3 text-sm leading-6 text-muted-foreground">
                  ...{result.context}...
                </p>
              ) : null}
            </Card>
          ))}
        </div>
      ) : null}

      {mode === "ask" ? (
        <Card className="p-5" aria-live="polite">
          {chatMessages.length === 0 ? (
            <div className="flex items-start gap-3 text-sm text-muted-foreground">
              <MessageSquareText className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
              <p>Ask a question, then keep going with follow-ups in the same chat.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {chatMessages.map((message, index) => (
                <div key={`${message.role}-${index}`} className="space-y-2">
                  <div className={message.role === "user" ? "text-right" : ""}>
                    <div
                      className={
                        message.role === "user"
                          ? "inline-block max-w-[85%] rounded-lg bg-primary px-4 py-3 text-sm text-primary-foreground"
                          : "inline-block max-w-[85%] rounded-lg bg-secondary px-4 py-3 text-sm text-secondary-foreground"
                      }
                    >
                      {message.content}
                    </div>
                  </div>
                  {message.role === "assistant" && message.sources && message.sources.length > 0 ? (
                    <div className="space-y-2">
                      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Sources</p>
                      <ul className="space-y-2">
                        {message.sources.map((source, sourceIndex) => (
                          <li
                            key={`${source.video_id}-${sourceIndex}`}
                            className="rounded-lg border border-border bg-muted/40 p-2 text-sm"
                          >
                            <Link href={`/videos/${source.video_id}`} className="font-medium text-primary hover:underline">
                              {source.video_title}
                            </Link>
                            {source.timestamp_formatted ? (
                              <Badge variant="muted" className="ml-2 align-middle">
                                {source.timestamp_formatted}
                              </Badge>
                            ) : null}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          )}

          {askMutation.isPending ? (
            <div className="mt-4 flex items-center gap-2 text-sm text-muted-foreground" role="status" aria-live="polite">
              <Spinner label="Thinking" />
              <span>Thinking...</span>
            </div>
          ) : null}

          {askMutation.isError ? (
            <ErrorState
              className="mt-4"
              title="Ask failed"
              message={getErrorMessage(askMutation.error, "The question could not be answered.")}
              onRetry={askMutation.variables ? retryAsk : undefined}
              retryLabel="Retry question"
            />
          ) : null}
        </Card>
      ) : null}
    </div>
  );
}
