import * as React from "react";

import { cn } from "@/lib/utils";

type FieldControlProps = {
  "aria-describedby"?: string;
  "aria-invalid"?: boolean | "false" | "true";
  id?: string;
};

export interface FieldProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "children"> {
  children: React.ReactElement<FieldControlProps>;
  description?: React.ReactNode;
  error?: React.ReactNode;
  id?: string;
  label: React.ReactNode;
}

export function Field({ children, className, description, error, id, label, ...props }: FieldProps) {
  const generatedId = React.useId();
  const control = React.Children.only(children);
  const controlId = id ?? control.props.id ?? generatedId;
  const descriptionId = description ? `${controlId}-description` : undefined;
  const errorId = error ? `${controlId}-error` : undefined;
  const describedBy = [control.props["aria-describedby"], descriptionId, errorId].filter(Boolean).join(" ");

  return (
    <div className={cn("space-y-2", className)} {...props}>
      <label htmlFor={controlId} className="block text-sm font-medium text-foreground">
        {label}
      </label>
      {React.cloneElement(control, {
        "aria-describedby": describedBy || undefined,
        "aria-invalid": error ? true : control.props["aria-invalid"],
        id: controlId,
      })}
      {description ? (
        <p id={descriptionId} className="text-xs text-muted-foreground">
          {description}
        </p>
      ) : null}
      {error ? (
        <p id={errorId} className="text-xs font-medium text-destructive">
          {error}
        </p>
      ) : null}
    </div>
  );
}
