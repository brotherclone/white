// web/components/StatusBadge.tsx
import { CandidateStatus } from "@/lib/types";

const styles: Record<CandidateStatus, string> = {
  approved: "bg-[#f4faec] text-[#3a6e00] border border-[#abd96d]",
  accepted: "bg-[#f4faec] text-[#3a6e00] border border-[#abd96d]",
  pending:  "bg-[#fffde8] text-[#7a6200] border border-[#ffff00]",
  rejected: "bg-[#fdf0f3] text-[#7a001c] border border-[#AE0A33]",
};

export default function StatusBadge({ status }: { status: CandidateStatus }) {
  return (
    <span
      className={`inline-block px-2 py-0.5 text-[0.7rem] lowercase tracking-[0.02em] ${
        styles[status] ?? "bg-[#f0f0f0] text-[#383838] border border-[#c0c0c0]"
      }`}
    >
      {status}
    </span>
  );
}
