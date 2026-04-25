export type CandidateStatus = "pending" | "approved" | "accepted" | "rejected";

export interface SongEntry {
  id: string;
  thread_slug: string;
  production_slug: string;
  production_path: string;
  title: string;
  key: string | null;
  bpm: number | null;
  rainbow_color: string | null;
  singer: string | null;
  has_decisions: boolean;
}

export interface Scores {
  composite: number;
  theory?: Record<string, number>;
  theory_total?: number;
  chromatic?: {
    match?: number;
    confidence?: number;
    temporal?: Record<string, number>;
    spatial?: Record<string, number>;
    ontological?: Record<string, number>;
  };
}

export interface Candidate {
  id: string;
  candidate_id: string;
  phase: string;
  section: string;
  template: string;
  status: CandidateStatus;
  rank: number;
  composite_score: number;
  midi_url: string;
  label: string;
  use_case: string;
  scores: Scores;
}
