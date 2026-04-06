export type CandidateStatus = "pending" | "approved" | "accepted" | "rejected";

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
  scores: Scores;
}
