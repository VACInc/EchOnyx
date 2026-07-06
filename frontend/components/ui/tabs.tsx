"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

export interface TabItem {
  content: React.ReactNode;
  disabled?: boolean;
  id: string;
  label: React.ReactNode;
}

export interface TabsProps {
  className?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
  tabs: TabItem[];
  value?: string;
}

export function Tabs({ className, defaultValue, onValueChange, tabs, value }: TabsProps) {
  const generatedId = React.useId();
  const firstEnabledTab = tabs.find((tab) => !tab.disabled)?.id;
  const [internalValue, setInternalValue] = React.useState(defaultValue ?? firstEnabledTab ?? "");
  const selectedValue = value ?? internalValue;

  const selectTab = (nextValue: string) => {
    if (value === undefined) setInternalValue(nextValue);
    onValueChange?.(nextValue);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const enabledTabs = tabs.filter((tab) => !tab.disabled);
    const selectedIndex = enabledTabs.findIndex((tab) => tab.id === selectedValue);
    if (selectedIndex === -1) return;

    const move = (offset: number) => {
      event.preventDefault();
      const nextIndex = (selectedIndex + offset + enabledTabs.length) % enabledTabs.length;
      const nextId = enabledTabs[nextIndex].id;
      selectTab(nextId);
      document.getElementById(`${generatedId}-tab-${nextId}`)?.focus();
    };

    if (event.key === "ArrowRight" || event.key === "ArrowDown") move(1);
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") move(-1);
    if (event.key === "Home") {
      event.preventDefault();
      selectTab(enabledTabs[0].id);
      document.getElementById(`${generatedId}-tab-${enabledTabs[0].id}`)?.focus();
    }
    if (event.key === "End") {
      event.preventDefault();
      const lastTab = enabledTabs[enabledTabs.length - 1];
      selectTab(lastTab.id);
      document.getElementById(`${generatedId}-tab-${lastTab.id}`)?.focus();
    }
  };

  const activeTab = tabs.find((tab) => tab.id === selectedValue) ?? tabs.find((tab) => !tab.disabled);

  return (
    <div className={cn("space-y-4", className)}>
      <div role="tablist" className="flex border-b border-border" onKeyDown={handleKeyDown}>
        {tabs.map((tab) => {
          const selected = tab.id === activeTab?.id;
          return (
            <button
              key={tab.id}
              id={`${generatedId}-tab-${tab.id}`}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`${generatedId}-panel-${tab.id}`}
              disabled={tab.disabled}
              tabIndex={selected ? 0 : -1}
              onClick={() => selectTab(tab.id)}
              className={cn(
                "inline-flex items-center border-b-2 px-3 py-2 text-sm font-medium transition",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                selected
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
                tab.disabled && "cursor-not-allowed opacity-55",
              )}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          id={`${generatedId}-panel-${tab.id}`}
          role="tabpanel"
          aria-labelledby={`${generatedId}-tab-${tab.id}`}
          hidden={tab.id !== activeTab?.id}
          tabIndex={0}
          className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
}
