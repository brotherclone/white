interface Props {
  value: number; // 0.0–1.0
  width?: number; // bar character count
}

export default function ScoreBar({ value, width = 10 }: Props) {
  const pct = Math.max(0, Math.min(1, value));
  const color =
    pct >= 0.5 ? "bg-green-500" : pct >= 0.3 ? "bg-yellow-500" : "bg-red-500";

  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-zinc-700 rounded-full overflow-hidden flex-shrink-0">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${pct * 100}%` }}
        />
      </div>
      <span className="text-zinc-400 text-xs tabular-nums w-10">{value.toFixed(3)}</span>
    </div>
  );
}
