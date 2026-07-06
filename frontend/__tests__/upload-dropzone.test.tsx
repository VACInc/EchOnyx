import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { UploadDropzone } from "@/components/upload-dropzone";
import { api } from "@/lib/api";
import { makeVideoFile, renderWithProviders } from "./test-utils";

vi.mock("@/lib/api", () => ({
  api: {
    createBatch: vi.fn(),
    getSettings: vi.fn(),
    uploadVideo: vi.fn(),
  },
}));

const mockedApi = vi.mocked(api);

function fileInput(container: HTMLElement) {
  const input = container.querySelector("input[type='file']");
  if (!(input instanceof HTMLInputElement)) {
    throw new Error("File input was not rendered");
  }
  return input;
}

describe("UploadDropzone", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedApi.getSettings.mockResolvedValue({ processing: {} } as Awaited<ReturnType<typeof api.getSettings>>);
    mockedApi.uploadVideo.mockImplementation(async (_file, _title, onProgress) => {
      onProgress?.({ loadedBytes: 50, totalBytes: 100, percent: 50 });
      return {} as Awaited<ReturnType<typeof api.uploadVideo>>;
    });
    mockedApi.createBatch.mockImplementation(async (_files, name, onProgress) => {
      onProgress?.({ loadedBytes: 100, totalBytes: 100, percent: 100 });
      return {
        id: "batch-1",
        name: name ?? null,
        status: "queued",
        total_videos: 2,
        completed_videos: 0,
        failed_videos: 0,
        progress: 0,
        created_at: "2026-01-01T00:00:00Z",
      };
    });
  });

  it("uploads a single file with a progress callback", async () => {
    const user = userEvent.setup();
    const file = makeVideoFile("solo.mp4");
    const { container } = renderWithProviders(<UploadDropzone />);

    await user.upload(fileInput(container), file);
    expect(await screen.findByText("solo.mp4")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Upload video" }));

    await waitFor(() => expect(mockedApi.uploadVideo).toHaveBeenCalledTimes(1));
    expect(mockedApi.uploadVideo).toHaveBeenCalledWith(file, undefined, expect.any(Function));
    expect(typeof mockedApi.uploadVideo.mock.calls[0]?.[2]).toBe("function");
  });

  it("creates one batch with all selected files", async () => {
    const user = userEvent.setup();
    const first = makeVideoFile("first.mp4");
    const second = makeVideoFile("second.mp4");
    const { container } = renderWithProviders(<UploadDropzone />);

    await user.upload(fileInput(container), [first, second]);
    expect(await screen.findByText("2 files selected")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Upload as batch" }));

    await waitFor(() => expect(mockedApi.createBatch).toHaveBeenCalledTimes(1));
    expect(mockedApi.createBatch).toHaveBeenCalledWith([first, second], expect.stringContaining("Batch upload"), expect.any(Function));
  });

  it("shows a failed single upload error and keeps the file available with remaining files", async () => {
    const user = userEvent.setup();
    const failed = makeVideoFile("failed.mp4");
    const remaining = makeVideoFile("remaining.mp4");
    mockedApi.uploadVideo.mockRejectedValueOnce(new Error("Disk full"));
    const { container } = renderWithProviders(<UploadDropzone />);

    await user.upload(fileInput(container), failed);
    await user.click(await screen.findByRole("button", { name: "Upload video" }));

    expect(await screen.findAllByText("Disk full")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Retry upload" })).toBeInTheDocument();

    await user.upload(fileInput(container), remaining);
    expect(await screen.findByText("2 files selected")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry upload" }));

    await waitFor(() => expect(mockedApi.createBatch).toHaveBeenCalledTimes(1));
    const selectedList = screen.getByRole("list");
    expect(within(selectedList).getByText("failed.mp4")).toBeInTheDocument();
    expect(within(selectedList).getByText("remaining.mp4")).toBeInTheDocument();
    expect(mockedApi.createBatch).toHaveBeenCalledWith([failed, remaining], expect.any(String), expect.any(Function));
  });
});
