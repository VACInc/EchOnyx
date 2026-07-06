"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckSquare, ListChecks, SearchX, Settings, Square, Tags, Trash2 } from "lucide-react";

import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { TagInput } from "@/components/ui/tag-input";
import { useToast } from "@/components/ui/toast";

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function TodoRowSkeleton() {
  return (
    <Card className="p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex min-w-0 flex-1 gap-3">
          <Skeleton className="mt-0.5 h-5 w-5 shrink-0 rounded" />
          <div className="min-w-0 flex-1 space-y-3">
            <Skeleton className="h-4 w-11/12" />
            <div className="flex flex-wrap gap-2">
              <Skeleton className="h-6 w-36 rounded-full" />
              <Skeleton className="h-6 w-20 rounded-full" />
              <Skeleton className="h-6 w-24 rounded-full" />
            </div>
          </div>
        </div>
        <Skeleton className="h-8 w-8 shrink-0 rounded-full" />
      </div>
    </Card>
  );
}

export default function TodosPage() {
  const queryClient = useQueryClient();
  const toast = useToast();
  const [status, setStatus] = useState<"open" | "completed" | "all">("open");
  const [sort, setSort] = useState<"updated_at" | "created_at" | "completed_at" | "video_title">("updated_at");
  const [search, setSearch] = useState("");
  const [tags, setTags] = useState<string[]>([]);

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const labelsQuery = useQuery({
    queryKey: ["video-labels"],
    queryFn: api.getVideoLabels,
  });

  const actionItemsEnabled = settingsQuery.data?.action_items.enabled !== false;

  const actionItemsQuery = useQuery({
    queryKey: ["action-items", { status, sort, search, tags }],
    queryFn: () => api.getActionItems({ status, sort, search, tags, pageSize: 100 }),
    enabled: actionItemsEnabled,
  });

  const allTodosQuery = useQuery({
    queryKey: ["action-items", "all-count"],
    queryFn: () => api.getActionItems({ status: "all", pageSize: 1 }),
    enabled: actionItemsEnabled,
  });

  const labelSuggestions = useMemo(
    () => labelsQuery.data?.labels.map((label) => label.name) ?? [],
    [labelsQuery.data],
  );

  const invalidate = () =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey: ["action-items"] }),
      queryClient.invalidateQueries({ queryKey: ["video"] }),
      queryClient.invalidateQueries({ queryKey: ["summary"] }),
    ]);

  const updateMutation = useMutation({
    mutationFn: ({ id, completed }: { id: string; completed: boolean }) => api.updateActionItem(id, { completed }),
    onSuccess: invalidate,
    onError: (error) => {
      toast({
        variant: "error",
        title: "Todo update failed",
        description: getErrorMessage(error, "The todo could not be updated."),
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteActionItem(id),
    onSuccess: invalidate,
    onError: (error) => {
      toast({
        variant: "error",
        title: "Todo delete failed",
        description: getErrorMessage(error, "The todo could not be deleted."),
      });
    },
  });

  const hasActiveFilters = status !== "all" || sort !== "updated_at" || search.trim().length > 0 || tags.length > 0;
  const knowsAnyTodos = allTodosQuery.isSuccess && allTodosQuery.data !== undefined;
  const hasAnyTodos = (allTodosQuery.data?.total ?? 0) > 0;
  const items = actionItemsQuery.data?.items ?? [];

  if (settingsQuery.data && !settingsQuery.data.action_items.enabled) {
    return (
      <EmptyState
        icon={<Settings className="h-6 w-6" aria-hidden="true" />}
        headline="Todos are disabled"
        hint="Action items can be enabled from Settings when you want videos to collect todos."
        action={
          <Link
            href="/settings"
            className="inline-flex h-10 items-center justify-center rounded-lg border border-border bg-card px-4 text-sm font-medium text-card-foreground shadow-sm transition hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            Open Settings
          </Link>
        }
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Todos</h1>
          <p className="text-sm text-muted-foreground">Action items collected from videos and manual notes.</p>
        </div>
        <Badge variant="muted">{actionItemsQuery.data?.total ?? 0} items</Badge>
      </div>

      <Card className="p-4">
        <div className="grid gap-3 lg:grid-cols-[140px_180px_minmax(0,1fr)_minmax(0,1.4fr)]">
          <div className="space-y-2">
            <label htmlFor="todo-status" className="block text-sm font-medium text-foreground">
              Status
            </label>
            <Select
              id="todo-status"
              value={status}
              onChange={(event) => setStatus(event.target.value as typeof status)}
            >
              <option value="open">Open</option>
              <option value="completed">Completed</option>
              <option value="all">All</option>
            </Select>
          </div>

          <div className="space-y-2">
            <label htmlFor="todo-sort" className="block text-sm font-medium text-foreground">
              Sort
            </label>
            <Select
              id="todo-sort"
              value={sort}
              onChange={(event) => setSort(event.target.value as typeof sort)}
            >
              <option value="updated_at">Recently updated</option>
              <option value="created_at">Newest</option>
              <option value="completed_at">Recently completed</option>
              <option value="video_title">Video title</option>
            </Select>
          </div>

          <div className="space-y-2">
            <label htmlFor="todo-search" className="block text-sm font-medium text-foreground">
              Text
            </label>
            <Input
              id="todo-search"
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Filter by todo text"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Tags className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              Labels
            </div>
            <TagInput
              value={tags}
              onChange={setTags}
              suggestions={labelSuggestions}
              placeholder={labelsQuery.isLoading ? "Loading labels..." : "Filter by labels"}
            />
          </div>
        </div>
      </Card>

      {actionItemsQuery.isLoading ? (
        <div className="space-y-3" aria-busy="true" aria-label="Loading todos">
          <TodoRowSkeleton />
          <TodoRowSkeleton />
          <TodoRowSkeleton />
        </div>
      ) : null}

      {actionItemsQuery.isError ? (
        <ErrorState
          title="Todos failed to load"
          message={getErrorMessage(actionItemsQuery.error, "Refresh to try loading todos again.")}
          onRetry={() => actionItemsQuery.refetch()}
        />
      ) : null}

      {!actionItemsQuery.isLoading && !actionItemsQuery.isError && items.length > 0 ? (
        <div className="space-y-3">
          {items.map((item) => (
            <Card key={item.id} className="p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="flex min-w-0 gap-3">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => updateMutation.mutate({ id: item.id, completed: !item.completed })}
                    className="mt-0.5 h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                    aria-label={item.completed ? "Mark todo open" : "Mark todo complete"}
                  >
                    {item.completed ? (
                      <CheckSquare className="h-5 w-5 text-success" aria-hidden="true" />
                    ) : (
                      <Square className="h-5 w-5" aria-hidden="true" />
                    )}
                  </Button>
                  <div className="min-w-0 space-y-2">
                    <p
                      className={
                        item.completed
                          ? "text-sm leading-6 text-muted-foreground line-through"
                          : "text-sm leading-6 text-card-foreground"
                      }
                    >
                      {item.text}
                    </p>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <Link href={`/videos/${item.video_id}`} className="font-medium text-primary hover:underline">
                        {item.video_title}
                      </Link>
                      <Badge variant="muted">{item.source}</Badge>
                      {item.labels.map((label) => (
                        <Badge key={`${item.id}-${label}`} variant="outline">
                          {label}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => deleteMutation.mutate(item.id)}
                  className="h-8 w-8 border border-destructive/25 p-0 text-destructive hover:bg-destructive/10 hover:text-destructive"
                  aria-label="Delete todo"
                >
                  <Trash2 className="h-4 w-4" aria-hidden="true" />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      ) : null}

      {!actionItemsQuery.isLoading && !actionItemsQuery.isError && items.length === 0 ? (
        hasAnyTodos || (!knowsAnyTodos && hasActiveFilters) ? (
          <EmptyState
            icon={<SearchX className="h-6 w-6" aria-hidden="true" />}
            headline="No todos match these filters"
            hint="Adjust the status, text, or label filters to widen the list."
          />
        ) : (
          <EmptyState
            icon={<ListChecks className="h-6 w-6" aria-hidden="true" />}
            headline="No todos yet"
            hint="Todos will appear here when action items are found or added for videos."
          />
        )
      ) : null}
    </div>
  );
}
