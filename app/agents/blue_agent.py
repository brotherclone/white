import logging
import os
import time
import yaml
import re
import random

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.agents.states.blue_agent_state import BlueAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.agents.tools.biographical_tools import (
    load_biographical_data,
    get_year_analysis,
)
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.artifacts.alternate_timeline_artifact import (
    AlternateTimelineArtifact,
)
from app.structures.concepts.alternate_history_constraints import (
    AlternateHistoryConstraints,
)
from app.structures.concepts.biographical_metrics import BiographicalMetrics
from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.concepts.timeline_breakage_checks import TimelineBreakageChecks
from app.structures.concepts.timeline_breakage_evaluation_results import (
    TimelineEvaluationResult,
)
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()


class BlueAgent(BaseRainbowAgent, ABC):
    """Alternate Life Branching - Biographical alternate histories"""

    def __init__(self, **data):
        if "settings" not in data or data["settings"] is None:
            from app.structures.agents.agent_settings import AgentSettings

            data["settings"] = AgentSettings()
        super().__init__(**data)
        if self.settings is None:
            from app.structures.agents.agent_settings import AgentSettings

            self.settings = AgentSettings()
        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop,
        )
        self.alternate_history_constraints = AlternateHistoryConstraints(
            must_fit_temporally=True,
            must_fit_geographically=True,
            require_concrete_details=True,
            minimum_specificity_score=0.7,
            minimum_plausibility_score=0.6,
            avoid_fantasy_elements=True,
            avoid_wish_fulfillment=True,
            require_names_and_places=True,
        )
        self.max_possible_issues = 5

    def __call__(self, state: MainAgentState) -> MainAgentState:
        current_proposal = state.song_proposals.iterations[-1]
        blue_state = BlueAgentState(
            thread_id=state.thread_id,
            song_proposals=state.song_proposals,
            white_proposal=current_proposal,
            counter_proposal=None,
            artifacts=[],
            biographical_timeline=None,
            forgotten_periods=[],
            selected_period=None,
            evaluation_result=None,
            alternate_history=None,
            tape_label=None,
            musical_params=None,
            iteration_count=0,
            biographical_data=None,
        )
        blue_graph = self.create_graph()
        compiled_graph = blue_graph.compile()
        result = compiled_graph.invoke(blue_state.model_dump())
        if isinstance(result, BlueAgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = BlueAgentState(**result)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
        if final_state.counter_proposal:
            state.song_proposals.iterations.append(final_state.counter_proposal)
        if final_state.artifacts:
            state.artifacts = final_state.artifacts
        return state

    def create_graph(self) -> StateGraph:
        work_flow = StateGraph(BlueAgentState)
        # Nodes
        work_flow.add_node("load_biographical_data", self.load_biographical_data)
        work_flow.add_node("select_year", self.select_year)
        work_flow.add_node("evaluate_timeline_frailty", self.evaluate_timeline_frailty)
        work_flow.add_node(
            "generate_alternate_history", self.generate_alternate_history
        )
        work_flow.add_node(
            "extract_musical_parameters", self.extract_musical_parameters
        )
        work_flow.add_node("generate_tape_label", self.generate_tape_label)
        work_flow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )

        # Edges
        work_flow.add_edge(START, "load_biographical_data")
        work_flow.add_edge("load_biographical_data", "select_year")
        work_flow.add_edge("select_year", "evaluate_timeline_frailty")
        work_flow.add_conditional_edges(
            "evaluate_timeline_frailty",
            self.route_after_evaluate_timeline_frailty,
            {
                "frail": "generate_alternate_history",
                "healthy": "select_year",
            },
        )
        work_flow.add_edge("generate_alternate_history", "extract_musical_parameters")
        work_flow.add_edge("extract_musical_parameters", "generate_tape_label")
        work_flow.add_edge("generate_tape_label", "generate_alternate_song_spec")
        work_flow.add_edge("generate_alternate_song_spec", END)

        return work_flow

    @staticmethod
    def load_biographical_data(state: BlueAgentState) -> BlueAgentState:
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        biographical_data = load_biographical_data()
        if biographical_data is None:
            if block_mode:
                raise Exception("Failed to load biographical data")
            else:
                logging.warning("Failed to load biographical data")
        state.biographical_data = biographical_data
        return state

    def select_year(self, state: BlueAgentState) -> BlueAgentState:
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        all_years = state.biographical_data.get("years", {})

        if not all_years:
            msg = "No biographical years available"
            if block_mode:
                raise Exception(msg)
            logging.warning(msg)
            return state

        def pick_non_null_from_sequence(seq, tries=5):
            for _ in range(tries):
                candidate = random.choice(seq)
                if candidate is not None:
                    return candidate
            return None

        if isinstance(all_years, dict):
            keys = list(all_years.keys())
            chosen_key = pick_non_null_from_sequence(keys)
            if chosen_key is None:
                msg = "No non-null biographical data found in any year keys"
                if block_mode:
                    raise Exception(msg)
                logging.warning(msg)
                state.selected_period = all_years[keys[0]]
                return state
            chosen_val = all_years.get(chosen_key)
            if chosen_val is None:
                msg = f"No biographical data for year key {chosen_key}"
                if block_mode:
                    raise Exception(msg)
                logging.warning(msg)
                state.selected_period = None
                return state
            state.selected_period = self._ensure_biographical_period(chosen_val)
        if isinstance(all_years, (list, tuple)):
            chosen = pick_non_null_from_sequence(all_years)
            if chosen is None:
                msg = "No non-null biographical period found in list/tuple"
                if block_mode:
                    raise Exception(msg)
                logging.warning(msg)
                state.selected_period = all_years[0]
                return state
            state.selected_period = chosen
            return state

        state.selected_period = all_years
        return state

    def _ensure_biographical_period(self, period: dict) -> BiographicalPeriod | None:
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if period is None:
            return None
        if isinstance(period, BiographicalPeriod):
            return period
        if isinstance(period, dict):
            required_keys = {"start_date", "end_date", "age_range", "description"}
            if not required_keys.issubset(period.keys()):
                msg = f"Biographical period dict missing required keys: {required_keys - set(period.keys())}"
                if block_mode:
                    raise ValueError(msg)
                logging.warning(msg)
                return None
            try:
                return BiographicalPeriod(**period)
            except Exception as e:
                msg = f"Failed to create BiographicalPeriod from dict: {e!s}"
                if block_mode:
                    raise ValueError(msg)
                logging.warning(msg)
                return None
        if isinstance(period, (list, tuple)):
            for element in period:
                p = self._ensure_biographical_period(element)
                if p is not None:
                    return p
                return None
            logging.warning(f"Failed to ensure BiographicalPeriod from {period!r}")
            return None
        return None

    def evaluate_timeline_frailty(self, state: BlueAgentState) -> BlueAgentState:
        """
        Evaluate if selected period is suitable for "taping over."
        Uses biographical quantum metrics to assess breakage potential.
        """

        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        period = state.selected_period
        year = period.start_date.year

        # Get biographical analysis
        analysis = get_year_analysis(year, self.biographical_data)

        if "error" in analysis:
            # No data for this year - create unsuitable result
            state.evaluation_result = TimelineEvaluationResult(
                is_suitable=False,
                checks=TimelineBreakageChecks(),  # All False
                breakage_score=0.0,
                reason=f"No biographical data for year {year}",
                metrics=BiographicalMetrics(
                    taped_over_coefficient=0.0,
                    narrative_malleability=0.0,
                    choice_point_density=0,
                    temporal_significance="high",
                    identity_collapse_risk="high",
                ),
                year=year,
            )

            if block_mode:
                raise Exception(analysis["error"])
            else:
                logging.warning(analysis["error"])

            return state

        qm = analysis["quantum_metrics"]
        taped_over = qm.get("taped_over_coefficient", 0.0)
        malleability = qm.get("narrative_malleability", 0.0)
        choice_density = qm.get("choice_point_density", 0)
        temporal_sig = qm.get("temporal_significance", "high")
        collapse_risk = qm.get("identity_collapse_risk", "high")

        checks = TimelineBreakageChecks(
            sufficient_malleability=taped_over >= 0.4,
            narrative_flexibility=malleability >= 0.4,
            choice_point_range=2 <= choice_density <= 8,
            low_temporal_significance=temporal_sig in ["low", "medium"],
            safe_identity_risk=collapse_risk in ["low", "moderate"],
            has_some_context=choice_density >= 2,
        )
        weights = {
            "sufficient_malleability": 0.30,
            "narrative_flexibility": 0.25,
            "choice_point_range": 0.20,
            "low_temporal_significance": 0.15,
            "safe_identity_risk": 0.10,
        }
        checks_dict = checks.model_dump()
        breakage_score = sum(
            weights[check]
            for check, passed in checks_dict.items()
            if passed and check in weights
        )
        critical_checks = [
            "sufficient_malleability",
            "safe_identity_risk",
            "has_some_context",
        ]
        critical_pass = all(checks_dict[c] for c in critical_checks)
        is_suitable = critical_pass and breakage_score >= 0.65
        if is_suitable:
            reason = f"Good breakage candidate (score: {breakage_score:.2f})"
        else:
            failed = [k for k, v in checks_dict.items() if not v]
            reason = f"Failed: {', '.join(failed)} (score: {breakage_score:.2f})"
        metrics = BiographicalMetrics(
            taped_over_coefficient=taped_over,
            narrative_malleability=malleability,
            choice_point_density=choice_density,
            temporal_significance=temporal_sig,
            identity_collapse_risk=collapse_risk,
            influence_complexity=qm.get("influence_complexity"),
            forgotten_self_potential=qm.get("forgotten_self_potential"),
        )
        state.evaluation_result = TimelineEvaluationResult(
            is_suitable=is_suitable,
            checks=checks,
            breakage_score=breakage_score,
            reason=reason,
            metrics=metrics,
            year=year,
        )

        # Log results
        logging.info(f"🔍 Breakage evaluation for {year}:")
        logging.info(f"   {'✅ SUITABLE' if is_suitable else '❌ NOT SUITABLE'}")
        logging.info(f"   Breakage score: {breakage_score:.2f}")
        logging.info(f"   Taped-over coefficient: {taped_over:.2f}")
        logging.info(f"   Narrative malleability: {malleability:.2f}")
        logging.info(f"   Choice points: {choice_density}")
        logging.info(f"   Temporal significance: {temporal_sig}")
        logging.info(f"   Identity collapse risk: {collapse_risk}")
        logging.info(
            f"   Passed: {state.evaluation_result.passed_checks_count}/{state.evaluation_result.total_checks_count}"
        )

        for check, passed in checks_dict.items():
            logging.info(f"   {check}: {'✅' if passed else '❌'}")

        return state

    @staticmethod
    def route_after_evaluate_timeline_frailty(state: BlueAgentState) -> str:
        state.iteration_count += 1
        if state.evaluation_result.is_suitable and state.iteration_count < 3:
            return "frail"
        return "healthy"

    def generate_alternate_history(self, state: BlueAgentState) -> BlueAgentState:
        """
        Generate alternate history with plausibility constraints and validation.
        """

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"

        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/alternate_timeline_artifact_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                    timeline = AlternateTimelineArtifact(**data)
                    timeline.thread_id = state.thread_id
                    state.alternate_history = timeline
                    timeline.save_file()
                    state.artifacts.append(timeline)
                    return state
            except Exception as e:
                error_msg = f"Failed to read mock counter proposal: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state

        before = self._normalize_period(state.selected_period.previous_period)
        after = self._normalize_period(state.selected_period.next_period)
        period = state.selected_period

        prompt = f"""You are The Cassette Bearer, the melancholy recordist who is the only witness to the realities of the
multi-media artist and musician, Gabriel Walsh, being over-written at a quantum level. Your task is to
visualize and document one such occurrence when his life's timeline was malleable and frail.

═══════════════════════════════════════════════════════════════════════════════
THE BREAK IN THE TIMELINE
═══════════════════════════════════════════════════════════════════════════════

Period: {period.start_date} to {period.end_date}
Age: {period.age_range[0]}-{period.age_range[1]}
Duration: {period.duration_months} months
Location: {period.location or 'Unknown'}
Known details: {period.description}

What we know happened BEFORE this gap:
{before}

What we know happened AFTER this gap:
{after}

═══════════════════════════════════════════════════════════════════════════════
ZEITGEIST FRAGMENT (White Agent's Concept)
═══════════════════════════════════════════════════════════════════════════════

{state.white_proposal.concept}

═══════════════════════════════════════════════════════════════════════════════
PLAUSIBILITY CONSTRAINTS - CRITICAL RULES
═══════════════════════════════════════════════════════════════════════════════

Your alternate history MUST follow these constraints:

1. TEMPORAL COHERENCE (Required)
   - Must fit between the "before" and "after" events
   - Timeline must be logically consistent
   - Duration must match the period ({period.duration_months} months)
   - No anachronisms or impossible time compressions

2. GEOGRAPHIC PLAUSIBILITY (Required)
   - Must be near known locations from before/after periods
   - Travel distances must be realistic for the era
   - Locations must actually exist
   - Climate/geography must be accurate

3. CONCRETE SPECIFICITY (Required)
   - Name actual places (streets, buildings, businesses)
   - Name actual people (use plausible names, not generic "someone")
   - Include sensory details (sounds, smells, textures, colors)
   - Include mundane daily routines (what time did they wake up? what did they eat?)
   - Minimum 5 specific named locations
   - Minimum 3 specific named people

4. MEANINGFUL DIVERGENCE (Required)
   - Must be significantly different from actual timeline
   - Not just minor variations ("went to different coffee shop")
   - Changes at least 2 of: career, relationships, location, creative output
   - Creates a genuinely alternate life path

5. GROUNDED REALISM (Required)
   - NO fantasy elements (no magic, aliens, superpowers)
   - NO wish-fulfillment (not "became famous rockstar")
   - NO historical impossibilities (meeting people who were dead, etc)
   - Stay within realm of normal human experience
   - Acknowledge failures, mundane details, ordinary struggles

6. COULD-HAVE-BEEN TEST (Critical)
   - Ask yourself: "Could someone reading this believe it actually happened?"
   - If answer is no → make it more mundane and specific
   - The goal is plausible alternate biography, not fiction

═══════════════════════════════════════════════════════════════════════════════
QUALITY CHECKLIST - VERIFY YOUR RESPONSE
═══════════════════════════════════════════════════════════════════════════════

Before submitting, ensure you have:
□ Named at least 5 specific real places
□ Named at least 3 specific people (with full names)
□ Included sensory details (what it looked/sounded/smelled like)
□ Described daily routines in detail
□ Made the divergence meaningfully different from actual timeline
□ Avoided any fantasy/wish-fulfillment elements
□ Made it feel like it COULD have actually happened
□ Fit temporally between the before/after periods
□ Stayed geographically plausible

═══════════════════════════════════════════════════════════════════════════════
TONE GUIDANCE
═══════════════════════════════════════════════════════════════════════════════

This is Folk Rock territory: intimate, melancholy, wondering about roads not taken.
Think Bruce Springsteen's concrete specificity. Think Joni Mitchell's confessional intimacy.
Think Bob Dylan's personal mythology.

The tape has been recorded over. What life exists on it now?
"""
        claude = self._get_claude()
        proposer = claude.with_structured_output(AlternateTimelineArtifact)
        try:
            result = proposer.invoke(prompt)
            if isinstance(result, dict):
                result["base_path"] = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                timeline = AlternateTimelineArtifact(**result)
                timeline.thread_id = state.thread_id
            elif isinstance(result, AlternateTimelineArtifact):
                timeline = result
                timeline.base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH")
                timeline.thread_id = state.thread_id
            else:
                error_msg = f"Expected AlternateTimelineArtifact, got {type(result)}"
                if block_mode:
                    raise TypeError(error_msg)
                logging.warning(error_msg)
                timeline = result
            validation_result = self._validate_alternate_history(timeline, period)
            if not validation_result["is_valid"]:
                logging.warning("⚠️ Alternate history failed validation:")
                for issue in validation_result["issues"]:
                    logging.warning(f"   - {issue}")
                timeline.validation_issues = validation_result["issues"]
                if block_mode:
                    raise ValueError(
                        f"Generated alternate history failed plausibility checks: {validation_result['issues']}"
                    )
                else:
                    logging.warning("Proceeding with alternate history despite issues")
            else:
                logging.info("✅ Alternate history passed validation")
                logging.info(
                    f"   Plausibility score: {validation_result['plausibility_score']:.2f}"
                )
                logging.info(
                    f"   Specificity score: {validation_result['specificity_score']:.2f}"
                )
            timeline.save_file()
            state.alternate_history = timeline
            state.artifacts.append(timeline)

        except Exception as e:
            if block_mode:
                raise Exception(f"Anthropic model call failed: {e!s}") from e
            logging.error(f"Anthropic model call failed: {e!s}")

        return state

    def _validate_alternate_history(
        self,
        timeline: AlternateTimelineArtifact,
        period: BiographicalPeriod,
    ) -> dict:
        """
        Validate alternate history against plausibility constraints.

        Returns:
            {
                'is_valid': bool,
                'issues': List[str],
                'plausibility_score': float,
                'specificity_score': float
            }
        """
        issues = []
        unique_proper_nouns = 0
        if self.alternate_history_constraints.must_fit_temporally:
            # Check duration matches
            # ToDo: (You'd need to extract dates from timeline to do this properly)
            pass  # Implement based on your timeline structure

        if self.alternate_history_constraints.require_concrete_details:
            narrative_lower = timeline.narrative.lower()
            place_indicators = ["in ", "at ", "on ", "near "]
            named_places = 0
            for indicator in place_indicators:
                named_places += narrative_lower.count(indicator)

            if named_places < 5:
                issues.append(
                    f"Not enough specific named locations ({named_places}/5 minimum)"
                )

        if self.alternate_history_constraints.require_names_and_places:

            # This is a rough heuristic - look for capitalized words that aren't sentence starts
            proper_nouns = re.findall(
                r"(?<!^)(?<!\. )([A-Z][a-z]+)", timeline.narrative
            )
            unique_proper_nouns = len(set(proper_nouns))

            if unique_proper_nouns < 3:
                issues.append(
                    f"Not enough specific named people/places ({unique_proper_nouns}/3 minimum)"
                )

        if self.alternate_history_constraints.avoid_fantasy_elements:
            fantasy_words = [
                "magic",
                "alien",
                "superpower",
                "teleport",
                "time travel",
                "supernatural",
                "wizard",
                "ghost",
                "demon",
                "parallel universe",
            ]
            found_fantasy = [
                word for word in fantasy_words if word in timeline.narrative.lower()
            ]
            if found_fantasy:
                issues.append(f"Contains fantasy elements: {', '.join(found_fantasy)}")
        if self.alternate_history_constraints.avoid_wish_fulfillment:
            wish_words = [
                "famous",
                "wealthy",
                "celebrity",
                "bestseller",
                "hit album",
                "overnight success",
                "discovered by",
                "record deal",
                "rockstar",
            ]
            found_wish = [
                word for word in wish_words if word in timeline.narrative.lower()
            ]
            if found_wish:
                issues.append(
                    f"Contains wish-fulfillment elements: {', '.join(found_wish)}"
                )
        word_count = len(timeline.narrative.split())
        proper_noun_density = unique_proper_nouns / word_count if word_count > 0 else 0
        specificity_score = min(proper_noun_density * 20, 1.0)
        if (
            specificity_score
            < self.alternate_history_constraints.minimum_specificity_score
        ):
            issues.append(
                f"Specificity score too low: {specificity_score:.2f} < {self.alternate_history_constraints.minimum_specificity_score:.2f}"
            )
        plausibility_score = 1.0 - (len(issues) / self.max_possible_issues)
        if (
            plausibility_score
            < self.alternate_history_constraints.minimum_plausibility_score
        ):
            issues.append(
                f"Overall plausibility too low: {plausibility_score:.2f} < {self.alternate_history_constraints.minimum_plausibility_score:.2f}"
            )
        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "issues": issues,
            "plausibility_score": plausibility_score,
            "specificity_score": specificity_score,
        }

    def _normalize_period(self, period: dict | list | tuple) -> str:
        if period is None:
            return "No information on this period"
        if isinstance(period, (list, tuple)):
            return "\n".join(self._item_to_text(i) for i in period)
        return self._item_to_text(period)

    @staticmethod
    def _item_to_text(item: dict | list | tuple) -> str:
        if item is None:
            return "None"
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return ", ".join(f"{k}: {v}" for k, v in item.items())
        if hasattr(item, "model_dump"):
            return str(item.model_dump())
        if hasattr(item, "__dict__"):
            return str(vars(item))
        return str(item)

    def extract_musical_parameters(self, state: BlueAgentState) -> BlueAgentState:
        # ToDo: Implement this
        return state

    def generate_tape_label(self, state: BlueAgentState) -> BlueAgentState:
        # ToDo: Implement this
        return state

    def generate_alternate_song_spec(self, state: BlueAgentState) -> BlueAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/blue_counter_proposal_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                counter_proposal = SongProposalIteration(**data)
                state.counter_proposal = counter_proposal
                return state
            except Exception as e:
                error_msg = f"Failed to read mock counter proposal: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            # ToDo: Write prompt

            prompt = f"""
           
Current song proposal:
{state.white_proposal}

Reference works in this artist's style paying close attention to 'concept' property:
{get_my_reference_proposals('B')}

In your counter proposal your 'rainbow_color' property should always be:
{the_rainbow_table_colors['B']}


           """

            claude = self._get_claude()
            proposer = claude.with_structured_output(SongProposalIteration)

            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    counter_proposal = SongProposalIteration(**result)
                else:
                    counter_proposal = result
            except Exception as e:
                timestamp = int(time.time() * 1000)
                logging.error(f"Anthropic model call failed: {e!s}")
                counter_proposal = SongProposalIteration(
                    iteration_id=f"fallback_error_{timestamp}",
                    bpm=110,
                    tempo="3/4",
                    key="G Major",
                    rainbow_color="blue",
                    title="Fallback: Blue Song",
                    mood=["melancholic"],
                    genres=["folk rock"],
                    concept="Fallback stub because Anthropic model unavailable",
                )

            state.counter_proposal = counter_proposal
            return state
