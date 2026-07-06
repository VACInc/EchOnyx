"use client";

import * as React from "react";

import { cn } from "@/lib/utils";

export interface TooltipProps {
  children: React.ReactElement<
    React.HTMLAttributes<HTMLElement> & {
      "aria-describedby"?: string;
    }
  >;
  className?: string;
  content: React.ReactNode;
  delayMs?: number;
}

export function Tooltip({ children, className, content, delayMs = 250 }: TooltipProps) {
  const tooltipId = React.useId();
  const [open, setOpen] = React.useState(false);
  const delayRef = React.useRef<number | null>(null);

  const clearDelay = React.useCallback(() => {
    if (delayRef.current !== null) {
      window.clearTimeout(delayRef.current);
      delayRef.current = null;
    }
  }, []);

  const show = React.useCallback(() => {
    clearDelay();
    delayRef.current = window.setTimeout(() => setOpen(true), delayMs);
  }, [clearDelay, delayMs]);

  const hide = React.useCallback(() => {
    clearDelay();
    setOpen(false);
  }, [clearDelay]);

  React.useEffect(() => clearDelay, [clearDelay]);

  const childProps = children.props;

  return (
    <span className="relative inline-flex" onMouseEnter={show} onMouseLeave={hide}>
      {React.cloneElement(children, {
        "aria-describedby": open ? [childProps["aria-describedby"], tooltipId].filter(Boolean).join(" ") : childProps["aria-describedby"],
        onBlur: (event: React.FocusEvent<HTMLElement>) => {
          childProps.onBlur?.(event);
          hide();
        },
        onFocus: (event: React.FocusEvent<HTMLElement>) => {
          childProps.onFocus?.(event);
          show();
        },
        onKeyDown: (event: React.KeyboardEvent<HTMLElement>) => {
          childProps.onKeyDown?.(event);
          if (event.key === "Escape") {
            hide();
          }
        },
      })}
      {open ? (
        <span
          id={tooltipId}
          role="tooltip"
          className={cn(
            "pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 max-w-xs -translate-x-1/2 rounded-md border border-border bg-card px-2 py-1 text-xs text-card-foreground shadow-lg",
            className,
          )}
        >
          {content}
        </span>
      ) : null}
    </span>
  );
}
