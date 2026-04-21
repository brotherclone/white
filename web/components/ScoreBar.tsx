// web/components/ScoreBar.tsx
interface Props {
  value: number; // 0.0–1.0
  trackWidth?: string; // CSS width, default "6rem"
}

export default function ScoreBar({ value, trackWidth = "6rem" }: Props) {
  const pct = Math.max(0, Math.min(1, value));
  // Brand palette: green (high) → orange (mid) → red (low)
  const fillColor =
    pct >= 0.5 ? "#abd96d" : pct >= 0.3 ? "#EF7143" : "#AE0A33";

  return (
    <div className="flex items-center gap-2">
      <div
        className="h-[4px] bg-[#d8d8d8] overflow-hidden flex-shrink-0"
        style={{ width: trackWidth }}
      >
        <div
          className="h-full transition-all duration-300"
          style={{ width: `${pct * 100}%`, backgroundColor: fillColor }}
        />
      </div>
      <span className="text-[#4b4b4b] text-[0.72rem] tabular-nums min-w-[2.5rem]">
        {value.toFixed(3)}
      </span>
    </div>
  );
}
