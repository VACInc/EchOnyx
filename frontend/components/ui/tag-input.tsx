"use client";

import * as React from "react";
import { X } from "lucide-react";

import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export interface TagInputProps {
  onChange: (value: string[]) => void;
  placeholder?: string;
  suggestions: string[];
  value: string[];
}

export function TagInput({ onChange, placeholder = "Add label", suggestions, value }: TagInputProps) {
  const inputId = React.useId();
  const listboxId = React.useId();
  const [inputValue, setInputValue] = React.useState("");
  const [activeIndex, setActiveIndex] = React.useState(0);
  const [listOpen, setListOpen] = React.useState(false);
  const selectedValues = React.useMemo(() => new Set(value.map((tag) => tag.toLowerCase())), [value]);

  const filteredSuggestions = React.useMemo(() => {
    const query = inputValue.trim().toLowerCase();
    return suggestions
      .filter((suggestion) => {
        const normalized = suggestion.toLowerCase();
        return !selectedValues.has(normalized) && (!query || normalized.includes(query));
      })
      .slice(0, 8);
  }, [inputValue, selectedValues, suggestions]);

  React.useEffect(() => {
    setActiveIndex(0);
  }, [inputValue]);

  const addTag = React.useCallback(
    (rawTag: string) => {
      const tag = rawTag.trim();
      if (!tag || selectedValues.has(tag.toLowerCase())) return;
      onChange([...value, tag]);
      setInputValue("");
      setListOpen(false);
    },
    [onChange, selectedValues, value],
  );

  const removeTag = (tagToRemove: string) => {
    onChange(value.filter((tag) => tag !== tagToRemove));
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" && filteredSuggestions.length > 0) {
      event.preventDefault();
      setListOpen(true);
      setActiveIndex((current) => (current + 1) % filteredSuggestions.length);
      return;
    }

    if (event.key === "ArrowUp" && filteredSuggestions.length > 0) {
      event.preventDefault();
      setListOpen(true);
      setActiveIndex((current) => (current - 1 + filteredSuggestions.length) % filteredSuggestions.length);
      return;
    }

    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      const suggestion = listOpen ? filteredSuggestions[activeIndex] : undefined;
      addTag(suggestion ?? inputValue);
      return;
    }

    if (event.key === "Backspace" && inputValue.length === 0 && value.length > 0) {
      event.preventDefault();
      onChange(value.slice(0, -1));
      return;
    }

    if (event.key === "Escape") {
      setListOpen(false);
    }
  };

  const showSuggestions = listOpen && filteredSuggestions.length > 0;

  return (
    <div className="relative">
      <div
        className={cn(
          "flex min-h-10 w-full flex-wrap items-center gap-2 rounded-lg border border-input bg-card px-3 py-2 shadow-sm transition",
          "focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 focus-within:ring-offset-background",
        )}
      >
        {value.map((tag) => (
          <span
            key={tag}
            className="inline-flex h-6 items-center gap-1 rounded-full bg-secondary px-2 text-xs font-medium text-secondary-foreground"
          >
            {tag}
            <button
              type="button"
              onClick={() => removeTag(tag)}
              className="rounded-full text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label={`Remove ${tag}`}
            >
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
        <Input
          id={inputId}
          value={inputValue}
          onChange={(event) => {
            setInputValue(event.target.value);
            setListOpen(true);
          }}
          onFocus={() => setListOpen(true)}
          onBlur={() => window.setTimeout(() => setListOpen(false), 100)}
          onKeyDown={handleKeyDown}
          placeholder={value.length === 0 ? placeholder : undefined}
          role="combobox"
          aria-autocomplete="list"
          aria-controls={listboxId}
          aria-expanded={showSuggestions}
          aria-activedescendant={showSuggestions ? `${listboxId}-${activeIndex}` : undefined}
          className="h-6 min-w-28 flex-1 border-0 bg-transparent p-0 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
        />
      </div>
      {showSuggestions ? (
        <div
          id={listboxId}
          role="listbox"
          className="absolute left-0 right-0 top-full z-30 mt-2 max-h-56 overflow-y-auto rounded-lg border border-border bg-card p-1 text-card-foreground shadow-lg"
        >
          {filteredSuggestions.map((suggestion, index) => (
            <button
              key={suggestion}
              id={`${listboxId}-${index}`}
              type="button"
              role="option"
              aria-selected={index === activeIndex}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => addTag(suggestion)}
              onMouseEnter={() => setActiveIndex(index)}
              className={cn(
                "flex w-full rounded-md px-3 py-2 text-left text-sm transition",
                index === activeIndex ? "bg-accent text-accent-foreground" : "hover:bg-accent hover:text-accent-foreground",
              )}
            >
              {suggestion}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
