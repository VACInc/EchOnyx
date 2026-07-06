import { useState } from "react";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { TagInput } from "@/components/ui/tag-input";
import { renderWithProviders } from "./test-utils";

function ControlledTagInput({ onChange = vi.fn() }: { onChange?: (value: string[]) => void }) {
  const [value, setValue] = useState(["alpha"]);

  return (
    <TagInput
      value={value}
      suggestions={["alpha", "beta", "gamma", "meeting notes"]}
      onChange={(next) => {
        setValue(next);
        onChange(next);
      }}
      placeholder="Add tag"
    />
  );
}

describe("TagInput", () => {
  it("renders chips and filters suggestions excluding selected tags", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ControlledTagInput />);

    expect(screen.getByText("alpha")).toBeInTheDocument();

    await user.type(screen.getByRole("combobox"), "a");

    const listbox = await screen.findByRole("listbox");
    expect(within(listbox).queryByRole("option", { name: "alpha" })).not.toBeInTheDocument();
    expect(within(listbox).getByRole("option", { name: "beta" })).toBeInTheDocument();
    expect(within(listbox).getByRole("option", { name: "gamma" })).toBeInTheDocument();
  });

  it("adds tags with Enter and removes them with backspace and click", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<ControlledTagInput onChange={onChange} />);
    const input = screen.getByRole("combobox");

    await user.type(input, "beta{Enter}");
    expect(onChange).toHaveBeenLastCalledWith(["alpha", "beta"]);
    expect(screen.getByText("beta")).toBeInTheDocument();

    await user.click(input);
    await user.keyboard("{Backspace}");
    expect(onChange).toHaveBeenLastCalledWith(["alpha"]);
    await waitFor(() => expect(screen.queryByText("beta")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Remove alpha" }));
    expect(onChange).toHaveBeenLastCalledWith([]);
    await waitFor(() => expect(screen.queryByText("alpha")).not.toBeInTheDocument());
  });
});
