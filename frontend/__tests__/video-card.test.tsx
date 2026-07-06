import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VideoCard } from "@/components/video-card";
import { api } from "@/lib/api";
import { renderWithProviders } from "./test-utils";

vi.mock("@/lib/api", () => ({
  api: {
    cancelJob: vi.fn(),
    resetVideo: vi.fn(),
    retryVideo: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

const baseVideo = {
  id: "video-1",
  original_filename: "demo.mp4",
  title: "Demo video",
  duration_formatted: "02:00",
  status: "processing",
  created_at: "2026-01-01T00:00:00Z",
  file_size: 1024,
  tags: ["demo"],
  duplicate_info: null,
};

describe("VideoCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedApi.cancelJob.mockResolvedValue({ message: "cancelled" });
  });

  it("shows status and duplicate badges", () => {
    renderWithProviders(
      <VideoCard
        video={{
          ...baseVideo,
          duplicate_info: {
            classification: "probable_duplicate",
            score: 0.91,
            suppressed: false,
            duplicate_of: { id: "video-0", title: "Original" },
          },
        }}
      />,
    );

    expect(screen.getByText("Processing")).toBeInTheDocument();
    expect(screen.getByText("Duplicate")).toBeInTheDocument();
  });

  it("shows Cancel for processing jobs and cancels after confirmation", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <VideoCard
        video={baseVideo}
        job={{
          id: "job-1",
          current_step: "summarization",
          progress: 42,
          step_progress: {
            summarization: {
              progress: 42,
              step_index: 7,
              step_count: 8,
            },
          },
        }}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("dialog", { name: "Cancel processing job?" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel job" }));

    await waitFor(() => expect(mockedApi.cancelJob).toHaveBeenCalledWith("job-1"));
    expect(await screen.findByText("Job cancelled")).toBeInTheDocument();
  });
});
