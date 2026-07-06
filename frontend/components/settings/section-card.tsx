import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface SectionCardProps {
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  description: string;
  icon?: LucideIcon;
  title: string;
}

export function SectionCard({ action, children, className, description, icon: Icon, title }: SectionCardProps) {
  return (
    <Card className={cn("p-5 sm:p-6", className)}>
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {Icon ? <Icon className="h-5 w-5 text-muted-foreground" aria-hidden="true" /> : null}
            <h2 className="text-lg font-semibold text-card-foreground">{title}</h2>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      <div className="mt-5">{children}</div>
    </Card>
  );
}
