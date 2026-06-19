"use client";

import { Collaborator, WorkOrder, WorkOrderStatus } from "@/lib/types";

// Rainbow palette — no cobalt blue anywhere
const STATUS_STYLES: Record<WorkOrderStatus, React.CSSProperties> = {
  draft:              { background: "#000",     color: "#cbcbcb", border: "1px solid #4b4b4b" },
  sent:               { background: "#042a7b",  color: "#ffff00", border: "1px solid #042a7b" },
  in_progress:        { background: "#EF7143",  color: "#000",    border: "1px solid #EF7143" },
  delivered:          { background: "#AD85D6",  color: "#000",    border: "1px solid #AD85D6" },
  accepted:           { background: "#abd96d",  color: "#383838", border: "1px solid #abd96d" },
  revision_requested: { background: "#AE0A33",  color: "#FCFCFC", border: "1px solid #AE0A33" },
};

const STATUS_LABELS: Record<WorkOrderStatus, string> = {
  draft:              "draft",
  sent:               "sent",
  in_progress:        "in progress",
  delivered:          "delivered",
  accepted:           "accepted",
  revision_requested: "revision",
};

function isOverdue(dateStr: string): boolean {
  return new Date(dateStr) < new Date(new Date().toDateString());
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

interface WorkOrderHudProps {
  workOrder: WorkOrder | null;
  collaborator: Collaborator | null;
  onOpen: () => void;
}

export default function WorkOrderHud({ workOrder, collaborator, onOpen }: WorkOrderHudProps) {
  if (!workOrder) {
    return (
      <div className="px-3 pb-2 flex flex-col gap-1.5">
        <p className="text-[10px] font-sans font-light text-[#4b4b4b]">No work order</p>
        <button
          onClick={onOpen}
          className="w-full py-1.5 text-[10px] font-sans bg-[#000] border border-[#333] text-[#9b9b9b] hover:text-[#f6f6f6] transition-colors"
          style={{ transitionDuration: "300ms", transitionTimingFunction: "ease-in-out" }}
        >
          create work order
        </button>
      </div>
    );
  }

  const overdue = workOrder.follow_up_date ? isOverdue(workOrder.follow_up_date) : false;

  return (
    <div className="px-3 pb-2 flex flex-col gap-1.5">
      {/* Collaborator name + status badge */}
      <div className="flex items-center justify-between gap-1">
        <span className="text-[10px] font-sans text-[#cbcbcb] truncate">
          {collaborator?.name ?? workOrder.collaborator_id}
        </span>
        <span
          className="text-[9px] font-sans px-1.5 py-0.5 flex-shrink-0"
          style={STATUS_STYLES[workOrder.status]}
        >
          {STATUS_LABELS[workOrder.status]}
        </span>
      </div>

      {/* Budget line */}
      {(workOrder.budget_agreed != null || workOrder.budget_paid != null) && (
        <p className="text-[9px] font-sans text-[#4b4b4b]">
          {workOrder.budget_agreed != null && (
            <span className="text-[#9b9b9b]">${workOrder.budget_agreed.toFixed(0)} agreed</span>
          )}
          {workOrder.budget_paid != null && workOrder.budget_paid > 0 && (
            <span> / ${workOrder.budget_paid.toFixed(0)} paid</span>
          )}
        </p>
      )}

      {/* Calendar chip — square, rainbow-colored */}
      {workOrder.follow_up_date && (
        <div
          className="flex items-center gap-1 text-[9px] font-sans px-1.5 py-0.5"
          style={
            overdue
              ? { background: "#AE0A33", color: "#FCFCFC" }
              : { background: "#042a7b", color: "#ffff00" }
          }
        >
          <span>📅</span>
          <span>
            {overdue ? "overdue " : "follow up "}
            {formatDate(workOrder.follow_up_date)}
            {workOrder.follow_up_reason && ` — ${workOrder.follow_up_reason}`}
          </span>
        </div>
      )}

      {/* Open button */}
      <button
        onClick={onOpen}
        className="w-full py-1.5 text-[10px] font-sans bg-[#000] border border-[#333] text-[#9b9b9b] hover:text-[#f6f6f6] transition-colors"
        style={{ transitionDuration: "300ms", transitionTimingFunction: "ease-in-out" }}
      >
        view work order
      </button>
    </div>
  );
}
