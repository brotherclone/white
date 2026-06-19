"use client";

import { Collaborator, WorkOrder, WorkOrderStatus } from "@/lib/types";

const STATUS_STYLES: Record<WorkOrderStatus, string> = {
  draft:              "bg-zinc-700 text-zinc-300 border-zinc-600",
  sent:               "bg-blue-900/50 text-blue-300 border-blue-700",
  in_progress:        "bg-yellow-900/40 text-yellow-300 border-yellow-700",
  delivered:          "bg-violet-900/40 text-violet-300 border-violet-700",
  accepted:           "bg-green-900/40 text-green-300 border-green-700",
  revision_requested: "bg-orange-900/40 text-orange-300 border-orange-700",
};

const STATUS_LABELS: Record<WorkOrderStatus, string> = {
  draft:              "Draft",
  sent:               "Sent",
  in_progress:        "In Progress",
  delivered:          "Delivered",
  accepted:           "Accepted",
  revision_requested: "Revision",
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
        <p className="text-[10px] font-sans text-zinc-600">No work order</p>
        <button
          onClick={onOpen}
          className="w-full py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200 transition-colors"
        >
          Create Work Order
        </button>
      </div>
    );
  }

  const overdue = workOrder.follow_up_date ? isOverdue(workOrder.follow_up_date) : false;

  return (
    <div className="px-3 pb-2 flex flex-col gap-1.5">
      {/* Collaborator name + status badge */}
      <div className="flex items-center justify-between gap-1">
        <span className="text-[10px] font-sans text-zinc-300 truncate">
          {collaborator?.name ?? workOrder.collaborator_id}
        </span>
        <span
          className={`text-[9px] font-sans px-1.5 py-0.5 rounded border flex-shrink-0 ${STATUS_STYLES[workOrder.status]}`}
        >
          {STATUS_LABELS[workOrder.status]}
        </span>
      </div>

      {/* Budget line */}
      {(workOrder.budget_agreed != null || workOrder.budget_paid != null) && (
        <p className="text-[9px] font-sans text-zinc-500">
          {workOrder.budget_agreed != null && (
            <span className="text-zinc-400">${workOrder.budget_agreed.toFixed(0)} agreed</span>
          )}
          {workOrder.budget_paid != null && workOrder.budget_paid > 0 && (
            <span> / ${workOrder.budget_paid.toFixed(0)} paid</span>
          )}
        </p>
      )}

      {/* Calendar chip */}
      {workOrder.follow_up_date && (
        <div
          className={`flex items-center gap-1 text-[9px] font-sans px-1.5 py-0.5 rounded ${
            overdue
              ? "bg-orange-900/30 text-orange-400 border border-orange-800"
              : "bg-blue-900/30 text-blue-400 border border-blue-800"
          }`}
        >
          <span>📅</span>
          <span>
            {overdue ? "Overdue " : "Follow up "}
            {formatDate(workOrder.follow_up_date)}
            {workOrder.follow_up_reason && ` — ${workOrder.follow_up_reason}`}
          </span>
        </div>
      )}

      {/* Open button */}
      <button
        onClick={onOpen}
        className="w-full py-1.5 text-[10px] font-sans rounded bg-zinc-800 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
      >
        View Work Order
      </button>
    </div>
  );
}
