import { Scores } from "@/lib/types";
import ScoreBar from "./ScoreBar";

function Row({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-zinc-500 text-xs w-36 flex-shrink-0 font-sans">{label}</span>
      <ScoreBar value={value} />
    </div>
  );
}

export default function ScorePanel({ scores }: { scores: Scores }) {
  const theory = scores.theory ?? {};
  const theoryTotal =
    scores.theory_total ??
    (Object.keys(theory).length > 0
      ? Object.values(theory).reduce((s, v) => s + v, 0) / Object.keys(theory).length
      : null);
  const chromatic = scores.chromatic ?? {};

  return (
    <div className="space-y-1.5">
      <Row label="Composite" value={scores.composite} />
      {theoryTotal !== null && <Row label="Theory (mean)" value={theoryTotal} />}
      {Object.entries(theory).map(([k, v]) => (
        <Row key={k} label={`  ${k}`} value={v} />
      ))}
      {chromatic.match !== undefined && <Row label="Chromatic match" value={chromatic.match} />}
      {chromatic.confidence !== undefined && (
        <div className="flex items-center gap-3">
          <span className="text-zinc-500 text-xs w-36 font-sans">Confidence</span>
          <span className="text-zinc-400 text-xs tabular-nums">{chromatic.confidence.toFixed(4)}</span>
        </div>
      )}
    </div>
  );
}
