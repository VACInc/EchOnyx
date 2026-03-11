// Dynamically determine API URL based on environment
function getApiUrl(): string {
  // If explicitly set, use that
  if (process.env.NEXT_PUBLIC_API_URL && process.env.NEXT_PUBLIC_API_URL !== "http://localhost:8000") {
    return process.env.NEXT_PUBLIC_API_URL;
  }

  // In browser, use same host with port 8000
  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    return `${protocol}//${hostname}:8000`;
  }

  // SSR fallback
  return "http://localhost:8000";
}

const API_URL = getApiUrl();

interface VideoResponse {
  id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  duration_seconds: number | null;
  duration_formatted: string;
  title: string | null;
  tags?: string[] | null;
  status: string;
  created_at: string;
}

interface VideoListResponse {
  videos: VideoResponse[];
  total: number;
  page: number;
  page_size: number;
}

interface VideoStatsResponse {
  total: number;
  completed: number;
  workload: number;
}

interface SettingsResponse {
  hardware_profile: string;
  gpu_backend: string;
  model_loading: string;
  models: {
    asr_family: string;
    asr_model: string;
    granite_force_cpu: boolean;
    diarization_model: string;
    vision_model: string;
    vision_mmproj: string;
    vision_chat_format: string;
    vision_endpoint_url: string;
    vision_endpoint_model: string;
    summarization_model: string;
    summarization_endpoint_url: string;
    summarization_endpoint_model: string;
    embedding_model: string;
    audio_event_model: string;
    rocm_llm_runtime: string;
    rocm_llm_idle_timeout_s: number;
  };
  runtime_planner: {
    enabled: boolean;
    gpu_memory_fraction: number;
    memory_ceiling_gb: number | null;
    accelerator_count: number;
    total_accelerator_memory_gb: number;
    effective_memory_budget_gb: number;
    placement_mode: string;
    worker_model_loading: string;
    keep_resident_models: string[];
    can_keep_all_worker_models_loaded: boolean;
    can_keep_endpoint_models_loaded: boolean;
    requires_endpoint_idle_teardown: boolean;
    endpoint_idle_timeout_recommendation_s: number;
    estimated_memory_by_model_gb: Record<string, number>;
    notes: string[];
  };
  processing: {
    max_video_length_hours: number;
    keyframe_extraction_interval: number;
    frame_persistence_seconds: number;
    frame_change_threshold: number;
    frame_stability_threshold: number;
    frame_dedupe_threshold: number;
    frame_resize_width: number;
    max_keyframes: number;
    min_speech_duration: number;
    batch_concurrent_jobs: number;
    summary_chunk_minutes: number;
    summary_chunk_overlap_minutes: number;
  };
}

interface SettingsUpdatePayload {
  asr_model?: string;
  runtime_planner_enabled?: boolean;
  gpu_memory_fraction?: number;
  runtime_memory_ceiling_gb?: number | null;
}

interface SummaryResponse {
  video_id: string;
  title: string | null;
  duration_formatted: string;
  speakers: string[];
  summary: {
    executive_summary: string;
    key_points: string[];
    action_items: string[];
    decisions: string[];
    topics: Array<{
      timestamp: string;
      topic: string;
      summary: string;
      speakers?: string[];
    }>;
  } | null;
  transcript: Array<{
    start: number;
    end: number;
    speaker: string | null;
    text: string;
  }>;
  slides: Array<{
    timestamp: number;
    image_path: string;
    ocr_text: string | null;
    description: string | null;
  }>;
}

interface SearchResult {
  video_id: string;
  video_title: string;
  timestamp: number | null;
  timestamp_formatted: string | null;
  speaker: string | null;
  text: string;
  context: string | null;
  relevance_score: number;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || "Request failed");
  }

  return res.json();
}

export const api = {
  // Videos
  async getVideos(params: { page?: number; pageSize?: number; search?: string }) {
    const query = new URLSearchParams();
    if (params.page) query.set("page", params.page.toString());
    if (params.pageSize) query.set("page_size", params.pageSize.toString());
    if (params.search) query.set("search", params.search);
    return fetchApi<VideoListResponse>(`/api/videos?${query}`);
  },

  async getVideoStats() {
    return fetchApi<VideoStatsResponse>("/api/videos/stats");
  },

  async getVideo(id: string) {
    return fetchApi<VideoResponse>(`/api/videos/${id}`);
  },

  async uploadVideo(file: File, title?: string) {
    const formData = new FormData();
    formData.append("file", file);
    if (title) formData.append("title", title);
    formData.append("auto_process", "true");

    const res = await fetch(`${API_URL}/api/videos/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(error.detail || "Upload failed");
    }

    return res.json() as Promise<VideoResponse>;
  },

  async deleteVideo(id: string) {
    return fetchApi<{ message: string }>(`/api/videos/${id}`, {
      method: "DELETE",
    });
  },

  async reprocessVideo(id: string) {
    return fetchApi<VideoResponse>(`/api/videos/${id}/reprocess`, {
      method: "POST",
    });
  },

  async resetVideo(id: string) {
    return fetchApi<VideoResponse>(`/api/videos/${id}/reset`, {
      method: "POST",
    });
  },

  async retryVideo(id: string) {
    return fetchApi<VideoResponse>(`/api/videos/${id}/retry`, {
      method: "POST",
    });
  },

  async updateVideoTags(id: string, tags: string[]) {
    return fetchApi<VideoResponse>(`/api/videos/${id}/tags`, {
      method: "PUT",
      body: JSON.stringify({ tags }),
    });
  },

  // Summaries
  async getSummary(videoId: string) {
    return fetchApi<SummaryResponse>(`/api/summaries/${videoId}`);
  },

  async exportSummary(videoId: string, format: "md" | "pdf" | "json") {
    const res = await fetch(
      `${API_URL}/api/summaries/${videoId}/export?format=${format}`
    );
    return res.blob();
  },

  // Search
  async search(query: string, videoId?: string, tags?: string[]) {
    const params = new URLSearchParams({ q: query });
    if (videoId) params.set("video_id", videoId);
    if (tags && tags.length > 0) {
      tags.forEach((tag) => params.append("tags", tag));
    }
    return fetchApi<SearchResponse>(`/api/search?${params}`);
  },

  async askQuestion(question: string, videoIds?: string[], tags?: string[]) {
    return fetchApi<{
      question: string;
      answer: string;
      sources: SearchResult[];
      confidence: number;
    }>("/api/search/ask", {
      method: "POST",
      body: JSON.stringify({ question, video_ids: videoIds, tags }),
    });
  },

  // Settings
  async getSettings() {
    return fetchApi<SettingsResponse>("/api/settings");
  },

  async updateSettings(payload: SettingsUpdatePayload) {
    return fetchApi<SettingsResponse>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  },

  async getAvailableModels() {
    return fetchApi<{
      asr: Array<{ name: string; size_gb: number; recommended: boolean }>;
    }>("/api/settings/models/available");
  },

  async getHardwareInfo() {
    return fetchApi<{
      nvidia_gpus: Array<{ name: string; vram_gb: number }>;
      amd_gpus: Array<{ name: string; vram_gb: number }>;
      unified_memory_gb: number | null;
      total_vram_gb: number;
      active_profile: string;
      active_backend: string;
      whisper_backend: string;
      asr_family: string;
      model_loading_strategy: string;
      rocm_llm_runtime: string;
      rocm_llm_idle_timeout_s: number;
      runtime_planner_enabled: boolean;
      runtime_memory_ceiling_gb: number | null;
      gpu_memory_fraction: number;
      runtime_plan: SettingsResponse["runtime_planner"];
    }>("/api/settings/hardware");
  },

  // Model Status
  async getModelStatus() {
    return fetchApi<{
      models: Record<string, {
        model_name: string;
        status: string;
        progress_percent?: number;
        speed_mbps?: number;
        eta_seconds?: number;
        file_size_gb?: number;
        expected_size_gb?: number;
        error?: string;
      }>;
      active_downloads: Array<{
        model_name: string;
        status: string;
        progress_percent: number;
        speed_mbps: number;
        eta_seconds: number | null;
      }>;
    }>("/api/settings/models/status");
  },

  // Jobs
  async getJobs(params: { status?: string; videoId?: string; page?: number; pageSize?: number }) {
    const query = new URLSearchParams();
    if (params.status) query.set("status", params.status);
    if (params.videoId) query.set("video_id", params.videoId);
    if (params.page) query.set("page", params.page.toString());
    if (params.pageSize) query.set("page_size", params.pageSize.toString());
    return fetchApi<{
      jobs: Array<{
        id: string;
        video_id: string;
        status: string;
        current_step: string | null;
        progress: number;
        step_progress: Record<string, {
          progress: number;
          eta_seconds?: number;
          step_index?: number;
          step_count?: number;
          started_at?: string;
          completed_at?: string;
          duration_seconds?: number;
        }> | null;
        error_message: string | null;
        error_step: string | null;
        started_at: string | null;
        completed_at: string | null;
        created_at: string;
      }>;
      total: number;
      page: number;
      page_size: number;
    }>(`/api/jobs?${query}`);
  },

  async getJob(id: string) {
    return fetchApi<{
      id: string;
      video_id: string;
      status: string;
      current_step: string | null;
      progress: number;
      step_progress: Record<string, {
        progress: number;
        eta_seconds?: number;
        step_index?: number;
        step_count?: number;
        started_at?: string;
        completed_at?: string;
        duration_seconds?: number;
      }> | null;
      error_message: string | null;
      error_step: string | null;
      started_at: string | null;
      completed_at: string | null;
      created_at: string;
    }>(`/api/jobs/${id}`);
  },

  async cancelOrphanedJobs() {
    return fetchApi<{ cancelled: number }>("/api/jobs/cancel-orphaned", {
      method: "POST",
    });
  },
};
