export type CandidateStatus = "pending" | "approved" | "accepted" | "rejected";

export interface SongEntry {
  id: string;
  thread_slug: string;
  production_slug: string;
  production_path: string;
  title: string;
  key: string | null;
  bpm: number | null;
  time_sig: string | null;
  rainbow_color: string | null;
  singer: string | null;
  schema_version: string;
  stub: boolean;
  initialized: boolean;
  has_mix: boolean;
  stage: "ideation" | "generation" | "composition" | "production" | "mixing" | "complete" | "invalid";
  proposal_path: string | null;
  concept: string | null;
}

export interface RegressionInfo {
  destructive: boolean;
  files_to_delete: string[];
}

export interface SampleEntry {
  segment_id: string;
  song_slug: string;
  color: string;
  match: number;
  audio_url: string;
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
  | "vocal_placeholders"
  | "recording"
  | "augmentation"
  | "cleaning"
  | "rough_mix"
  | "mix_candidate"
  | "final_mix";

export const MIX_STAGES: MixStage[] = [
  "structure", "lyrics", "vocal_placeholders", "recording",
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

export interface LyricCandidate {
  id: string;
  rank: number;
  status: "pending" | "approved" | "promoted";
  text: string;
  match: number | null;
  fitting_verdict: string | null;
}

export interface LyricsResponse {
  status: "pending" | "promoted";
  candidates: LyricCandidate[];
}

export interface Scores {
  composite?: number;
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

// ---------------------------------------------------------------------------
// Diary
// ---------------------------------------------------------------------------

export interface DiaryEntry {
  id: string;
  song_slug: string;
  phase: string | null;
  author: string;
  created_at: string;
  title: string | null;
  body: string;
  tags: string[];
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Collaborator registry
// ---------------------------------------------------------------------------

export type CollaboratorRole =
  | "vocalist" | "drummer" | "guitarist" | "bassist"
  | "keys" | "strings" | "brass" | "mixing" | "mastering" | "other";

export type CollaboratorPlatform = "airgigs" | "soundbetter" | "direct" | "other";

export type PROAffiliation = "ascap" | "bmi" | "sesac" | "socan" | "prs" | "other" | "none";

export interface AvailabilityWindow {
  unavailable_from: string;  // ISO date "YYYY-MM-DD"
  unavailable_until: string;
  reason?: string | null;
}

export interface PlatformProfile {
  platform: CollaboratorPlatform;
  url: string;
  username?: string | null;
}

export interface Collaborator {
  id: string;
  name: string;
  roles: CollaboratorRole[];
  email?: string | null;
  photo_url?: string | null;
  platforms: PlatformProfile[];
  website?: string | null;
  socials: Record<string, string>;
  pro_affiliation: PROAffiliation;
  pro_number?: string | null;
  availability_windows: AvailabilityWindow[];
  notes: string;
}

// ---------------------------------------------------------------------------
// Work orders
// ---------------------------------------------------------------------------

export type WorkOrderStatus =
  | "draft" | "sent" | "in_progress" | "delivered" | "accepted" | "revision_requested";

export type BudgetStatus = "pending" | "agreed" | "invoiced" | "paid";

export interface RoyaltySplit {
  collaborator_id: string;
  song_slug: string;
  mechanical_pct: number;
  performance_pct: number;
  sync_pct: number;
  notes: string;
}

export interface WorkOrder {
  id: string;
  song_slug: string;
  collaborator_id: string;
  role: CollaboratorRole;
  platform: CollaboratorPlatform;
  status: WorkOrderStatus;
  key: string;
  bpm?: number | null;
  time_signature: string;
  sections: string[];
  creative_direction: string;
  part_notes: string;
  artifact_types: string[];
  deliverable_format: string;
  deadline?: string | null;
  follow_up_date?: string | null;
  follow_up_reason: string;
  budget_agreed?: number | null;
  budget_paid?: number | null;
  budget_currency: string;
  budget_status: BudgetStatus;
  calendar_event_id?: string | null;
  royalty_split?: RoyaltySplit | null;
  created_at: string;
  updated_at: string;
}
