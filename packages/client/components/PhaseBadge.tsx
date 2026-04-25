// web/components/PhaseBadge.tsx  (new component)
//
// Colour-codes each generation phase using the Rainbow Table palette.
// chords → black  (album I)
// drums  → red    (album II)
// bass   → orange (album III)
// melody → green  (album V)
// quartet→ indigo (album VII)

type Phase = "chords" | "drums" | "bass" | "melody" | "quartet";

const PHASE_STYLES: Record<Phase, { bg: string; fg: string }> = {
  chords:  { bg: "#000000", fg: "#f6f6f6" },
  drums:   { bg: "#AE0A33", fg: "#FCFCFC" },
  bass:    { bg: "#EF7143", fg: "#000000" },
  melody:  { bg: "#abd96d", fg: "#383838" },
  quartet: { bg: "#26294a", fg: "#F4A78A" },
};

export default function PhaseBadge({ phase }: { phase: string }) {
  const s = PHASE_STYLES[phase as Phase] ?? { bg: "#cbcbcb", fg: "#000" };
  return (
    <span
      className="inline-block px-[0.45rem] py-[0.15rem] text-[0.6rem] font-bold uppercase tracking-[0.08em]"
      style={{ fontFamily: "anisette-std, sans-serif", backgroundColor: s.bg, color: s.fg }}
    >
      {phase}
    </span>
  );
}
