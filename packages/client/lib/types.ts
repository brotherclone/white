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
  initialized: boolean;
  proposal_path: string | null;
}

export interface PipelineStatus {
  initialized: boolean;
  phases: Record<string, string>;
  next_phase: string | null;
  phase_order: string[];
}

export type MixStage =
  | "structure"
  | "lyrics"
  | "recording"
  | "vocal_placeholders"
  | "augmentation"
  | "cleaning"
  | "rough_mix"
  | "mix_candidate"
  | "final_mix";

export const MIX_STAGES: MixStage[] = [
  "structure", "lyrics", "recording", "vocal_placeholders",
  "augmentation", "cleaning", "rough_mix", "mix_candidate", "final_mix",
];

export interface CompositionVersion {
  version: number;
  created: string;
  stage: MixStage;
  notes: string;
}

export interface CompositionEntry {
  song_title: string;
  thread_slug: string;
  production_slug: string;
  logic_project_path: string;
  current_version: number;
  current_stage: MixStage;
  versions: CompositionVersion[];
}

export interface RunJob {
  status: "idle" | "running" | "done" | "error";
  phase: string | null;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
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
