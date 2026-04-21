// web/components/ScorePanel.tsx
import { Scores } from "@/lib/types";
import ScoreBar from "./ScoreBar";

function Row({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-3">
      <span
        className="text-[#4b4b4b] text-[0.72rem] w-36 flex-shrink-0 font-light"
        style={{ fontFamily: "semplicitapro, sans-serif" }}
      >
        {label}
      </span>
      <ScoreBar value={value} />
    </div>
  );
}

export default function ScorePanel({ scores }: { scores: Scores }) {
  const theory = scores.theory ?? {};
  const theoryTotal =
    scores.theory_total ??
    (Object.keys(theory).length > 0
      ? Object.values(theory).reduce((s, v) => s + v, 0) /
        Object.keys(theory).length
      : null);
  const chromatic = scores.chromatic ?? {};

  return (
    <div className="space-y-1.5 py-2">
      <Row label="composite" value={scores.composite} />
      {theoryTotal !== null && <Row label="theory (mean)" value={theoryTotal} />}
      {Object.entries(theory).map(([k, v]) => (
        <Row key={k} label={`  ${k.replace(/_/g, " ")}`} value={v} />
      ))}
      {chromatic.match !== undefined && (
        <Row label="chromatic match" value={chromatic.match} />
      )}
      {chromatic.confidence !== undefined && (
        <div className="flex items-center gap-3">
          <span
            className="text-[#4b4b4b] text-[0.72rem] w-36 font-light"
            style={{ fontFamily: "semplicitapro, sans-serif" }}
          >
            confidence
          </span>
          <span className="text-[#4b4b4b] text-[0.72rem] tabular-nums">
            {chromatic.confidence.toFixed(4)}
          </span>
        </div>
      )}
    </div>
  );
}
