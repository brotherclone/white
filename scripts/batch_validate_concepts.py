# scripts/batch_validate_concepts.py

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging
from dataclasses import dataclass, asdict

from your_white_agent import create_white_agent_graph  # Your actual import


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class BatchConfig:
    """Configuration for batch validation run"""

    total_runs: int = 100
    output_dir: Path = Path("validation_data")
    checkpoint_every: int = 10  # Save progress every N runs
    log_level: str = "INFO"
    resume_from_run: int | None = None  # Set to resume from specific run
    save_all_proposals: bool = True  # Harvest proposal iterations

    # API rate limiting
    delay_between_runs: float = 1.0  # Seconds between runs
    max_retries: int = 3
    retry_delay: float = 60.0  # Seconds to wait on retry


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ProposalIteration:
    """Single proposal iteration from song proposal generation"""

    iteration: int
    corpus_selections: List[str]
    rebracketing_method: str
    concept_text: str
    quality_score: float
    selected: bool
    timestamp: str


@dataclass
class ValidationRun:
    """Complete validation run with all data"""

    run_id: str
    run_number: int
    start_time: str
    end_time: str | None
    duration_seconds: float | None
    status: str  # "success", "failed", "partial"

    # Proposal data
    all_proposals: List[ProposalIteration]
    selected_proposal_index: int | None

    # Cascade data (only if selected proposal ran through full cascade)
    agents_executed: List[str]
    artifacts_generated: Dict[str, int]  # agent_name -> count
    final_synthesis: str | None

    # Diagnostics
    error_message: str | None
    token_usage: Dict[str, int] | None
    warnings: List[str]


@dataclass
class BatchResults:
    """Overall batch execution results"""

    batch_id: str
    start_time: str
    end_time: str | None
    total_runs: int
    completed_runs: int
    successful_runs: int
    failed_runs: int
    partial_runs: int

    runs: List[ValidationRun]

    # Aggregated metrics
    total_proposals_collected: int
    total_artifacts_generated: int
    agent_execution_counts: Dict[str, int]
    average_run_duration: float | None


# ============================================================================
# BATCH EXECUTOR
# ============================================================================


class BatchValidator:
    """Handles batch execution of validation runs"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.setup_logging()
        self.setup_output_dirs()
        self.batch_results = self.initialize_batch()

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=self.config.log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    self.config.output_dir
                    / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_output_dirs(self):
        """Create necessary output directories"""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "runs").mkdir(exist_ok=True)
        (self.config.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.config.output_dir / "proposals").mkdir(exist_ok=True)

    def initialize_batch(self) -> BatchResults:
        """Initialize or resume batch results"""
        if self.config.resume_from_run:
            return self.load_checkpoint(self.config.resume_from_run)

        return BatchResults(
            batch_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat(),
            end_time=None,
            total_runs=self.config.total_runs,
            completed_runs=0,
            successful_runs=0,
            failed_runs=0,
            partial_runs=0,
            runs=[],
            total_proposals_collected=0,
            total_artifacts_generated=0,
            agent_execution_counts={},
            average_run_duration=None,
        )

    def run_single_validation(self, run_number: int) -> ValidationRun:
        """Execute a single validation run"""
        run_id = f"run_{self.batch_results.batch_id}_{run_number:04d}"
        start_time = datetime.now()

        self.logger.info(f"Starting {run_id} ({run_number}/{self.config.total_runs})")

        try:
            # Create fresh graph for this run
            graph = create_white_agent_graph()

            # Initial state (let White Agent generate concept)
            initial_state = {
                "messages": [],
                "concept": None,
                "all_proposals": [],  # HARVEST ITERATIONS HERE
                "selected_proposal": None,
                # ... other initial state
            }

            # Execute workflow
            config = {"configurable": {"thread_id": run_id}}

            final_state = None
            for event in graph.stream(initial_state, config, stream_mode="values"):
                final_state = event

                # Optionally log progress
                if "current_agent" in event:
                    self.logger.debug(f"{run_id}: {event['current_agent']}")

            # Extract results
            validation_run = self.extract_validation_data(
                run_id, run_number, start_time, final_state
            )

            self.logger.info(
                f"✅ {run_id} completed: "
                f"{len(validation_run.all_proposals)} proposals, "
                f"{len(validation_run.agents_executed)} agents"
            )

            return validation_run

        except Exception as e:
            self.logger.error(f"❌ {run_id} failed: {str(e)}")
            return ValidationRun(
                run_id=run_id,
                run_number=run_number,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                status="failed",
                all_proposals=[],
                selected_proposal_index=None,
                agents_executed=[],
                artifacts_generated={},
                final_synthesis=None,
                error_message=str(e),
                token_usage=None,
                warnings=[],
            )

    def extract_validation_data(
        self,
        run_id: str,
        run_number: int,
        start_time: datetime,
        final_state: Dict[str, Any],
    ) -> ValidationRun:
        """Extract structured data from workflow execution"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Extract all proposal iterations
        all_proposals = []
        if self.config.save_all_proposals and "all_proposals" in final_state:
            for i, prop_data in enumerate(final_state["all_proposals"]):
                all_proposals.append(
                    ProposalIteration(
                        iteration=i + 1,
                        corpus_selections=prop_data.get("corpus_selections", []),
                        rebracketing_method=prop_data.get(
                            "rebracketing_method", "unknown"
                        ),
                        concept_text=prop_data.get("concept_text", ""),
                        quality_score=prop_data.get("quality_score", 0.0),
                        selected=prop_data.get("selected", False),
                        timestamp=prop_data.get("timestamp", ""),
                    )
                )

        # Extract cascade execution data
        agents_executed = final_state.get("agents_executed", [])
        artifacts = final_state.get("artifacts", {})

        artifacts_count = {}
        for agent, artifact_list in artifacts.items():
            artifacts_count[agent] = (
                len(artifact_list) if isinstance(artifact_list, list) else 1
            )

        # Determine status
        status = "success"
        if not agents_executed:
            status = "failed"
        elif len(agents_executed) < 8:
            status = "partial"

        return ValidationRun(
            run_id=run_id,
            run_number=run_number,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            status=status,
            all_proposals=all_proposals,
            selected_proposal_index=final_state.get("selected_proposal_index"),
            agents_executed=agents_executed,
            artifacts_generated=artifacts_count,
            final_synthesis=final_state.get("final_synthesis"),
            error_message=None,
            token_usage=final_state.get("token_usage"),
            warnings=final_state.get("warnings", []),
        )

    def save_run_data(self, run: ValidationRun):
        """Save individual run data"""
        # Save full run data
        run_file = self.config.output_dir / "runs" / f"{run.run_id}.json"
        with open(run_file, "w") as f:
            json.dump(asdict(run), f, indent=2)

        # Save proposals separately (easier to load for training)
        if run.all_proposals:
            proposals_file = (
                self.config.output_dir / "proposals" / f"{run.run_id}_proposals.json"
            )
            with open(proposals_file, "w") as f:
                json.dump([asdict(p) for p in run.all_proposals], f, indent=2)

    def save_checkpoint(self):
        """Save batch progress checkpoint"""
        checkpoint_file = (
            self.config.output_dir
            / "checkpoints"
            / f"checkpoint_{self.batch_results.completed_runs:04d}.json"
        )
        with open(checkpoint_file, "w") as f:
            json.dump(asdict(self.batch_results), f, indent=2)

        self.logger.info(
            f"💾 Checkpoint saved at run {self.batch_results.completed_runs}"
        )

    def load_checkpoint(self, run_number: int) -> BatchResults:
        """Load batch progress from checkpoint"""
        checkpoint_file = (
            self.config.output_dir / "checkpoints" / f"checkpoint_{run_number:04d}.json"
        )

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        with open(checkpoint_file) as f:
            data = json.load(f)

        self.logger.info(f"📂 Resuming from checkpoint at run {run_number}")
        return BatchResults(**data)

    def update_batch_metrics(self):
        """Update aggregated batch metrics"""
        self.batch_results.total_proposals_collected = sum(
            len(run.all_proposals) for run in self.batch_results.runs
        )

        self.batch_results.total_artifacts_generated = sum(
            sum(run.artifacts_generated.values()) for run in self.batch_results.runs
        )

        # Count agent executions
        agent_counts = {}
        for run in self.batch_results.runs:
            for agent in run.agents_executed:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        self.batch_results.agent_execution_counts = agent_counts

        # Calculate average duration
        durations = [
            run.duration_seconds
            for run in self.batch_results.runs
            if run.duration_seconds
        ]
        if durations:
            self.batch_results.average_run_duration = sum(durations) / len(durations)

    def print_progress(self):
        """Print progress summary"""
        print("\n" + "=" * 80)
        print("📊 BATCH VALIDATION PROGRESS")
        print("=" * 80)
        print(
            f"Completed: {self.batch_results.completed_runs}/{self.batch_results.total_runs}"
        )
        print(f"Successful: {self.batch_results.successful_runs}")
        print(f"Failed: {self.batch_results.failed_runs}")
        print(f"Partial: {self.batch_results.partial_runs}")
        print(f"Proposals Collected: {self.batch_results.total_proposals_collected}")
        print(f"Artifacts Generated: {self.batch_results.total_artifacts_generated}")

        if self.batch_results.average_run_duration:
            avg_mins = self.batch_results.average_run_duration / 60
            remaining_runs = (
                self.batch_results.total_runs - self.batch_results.completed_runs
            )
            eta_mins = avg_mins * remaining_runs
            print(f"Average Run Time: {avg_mins:.1f} minutes")
            print(
                f"Estimated Time Remaining: {eta_mins:.1f} minutes ({eta_mins / 60:.1f} hours)"
            )

        print("=" * 80 + "\n")

    def run_batch(self):
        """Execute the full batch of validation runs"""
        start_run = self.config.resume_from_run or 1

        for run_number in range(start_run, self.config.total_runs + 1):
            # Execute run
            validation_run = self.run_single_validation(run_number)

            # Update results
            self.batch_results.runs.append(validation_run)
            self.batch_results.completed_runs += 1

            if validation_run.status == "success":
                self.batch_results.successful_runs += 1
            elif validation_run.status == "failed":
                self.batch_results.failed_runs += 1
            elif validation_run.status == "partial":
                self.batch_results.partial_runs += 1

            # Save run data
            self.save_run_data(validation_run)

            # Checkpoint periodically
            if run_number % self.config.checkpoint_every == 0:
                self.update_batch_metrics()
                self.save_checkpoint()
                self.print_progress()

            # Rate limiting
            if run_number < self.config.total_runs:
                time.sleep(self.config.delay_between_runs)

        # Final save
        self.batch_results.end_time = datetime.now().isoformat()
        self.update_batch_metrics()
        self.save_final_results()
        self.print_final_summary()

    def save_final_results(self):
        """Save final batch results"""
        results_file = (
            self.config.output_dir / f"batch_results_{self.batch_results.batch_id}.json"
        )
        with open(results_file, "w") as f:
            json.dump(asdict(self.batch_results), f, indent=2)

        self.logger.info(f"📝 Final results saved to {results_file}")

    def print_final_summary(self):
        """Print final batch summary"""
        print("\n" + "=" * 80)
        print("🎉 BATCH VALIDATION COMPLETE")
        print("=" * 80)
        print(f"Batch ID: {self.batch_results.batch_id}")
        print(f"Total Runs: {self.batch_results.total_runs}")
        print(
            f"Successful: {self.batch_results.successful_runs} ({self.batch_results.successful_runs / self.batch_results.total_runs * 100:.1f}%)"
        )
        print(f"Failed: {self.batch_results.failed_runs}")
        print(f"Partial: {self.batch_results.partial_runs}")
        print("\n📊 Data Collected:")
        print(f"Total Proposals: {self.batch_results.total_proposals_collected}")
        print(f"Total Artifacts: {self.batch_results.total_artifacts_generated}")
        print("\n🎨 Agent Execution Counts:")
        for agent, count in sorted(self.batch_results.agent_execution_counts.items()):
            print(f"  {agent}: {count}")
        print(
            f"\n⏱️  Average Run Time: {self.batch_results.average_run_duration / 60:.1f} minutes"
        )
        print("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    config = BatchConfig(
        total_runs=100,
        output_dir=Path("validation_data"),
        checkpoint_every=10,
        save_all_proposals=True,
        delay_between_runs=1.0,
    )

    validator = BatchValidator(config)
    validator.run_batch()


if __name__ == "__main__":
    main()
