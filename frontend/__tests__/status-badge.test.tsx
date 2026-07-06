import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge } from "@/components/ui/status-badge";

describe("StatusBadge", () => {
  it.each([
    ["queued", "Queued"],
    ["processing", "Processing"],
    ["completed", "Completed"],
    ["failed", "Failed"],
    ["offline", "Offline"],
    ["mystery", "Unknown"],
  ])("maps %s to %s", (status, label) => {
    render(<StatusBadge status={status} />);

    expect(screen.getByText(label)).toBeInTheDocument();
  });
});
