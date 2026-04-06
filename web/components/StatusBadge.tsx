import { CandidateStatus } from "@/lib/types";

const styles: Record<CandidateStatus, string> = {
  approved: "bg-green-900/60 text-green-300 border-green-700",
  accepted: "bg-green-900/60 text-green-300 border-green-700",
  pending:  "bg-yellow-900/40 text-yellow-300 border-yellow-700",
  rejected: "bg-red-900/40 text-red-400 border-red-800",
};

export default function StatusBadge({ status }: { status: CandidateStatus }) {
  return (
    <span className={`inline-block px-2 py-0.5 text-xs rounded border font-sans ${styles[status] ?? "bg-zinc-800 text-zinc-400 border-zinc-600"}`}>
      {status}
    </span>
  );
}
