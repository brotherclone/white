import datetime
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

from app.reference.biographical.places_frequented import (
    EUROPE_LOCATIONS,
    US_LOCATIONS,
    NEW_JERSEY_LOCATIONS,
    NEW_YORK_LOCATIONS,
    DMV_LOCATIONS,
    ALL_LOCATIONS,
)
from app.reference.biographical.ytr_lyrics import YOUR_TEAM_RING_TAPE_LYRIC_FRAGMENTS
from app.reference.music.blue_agent_instruments import (
    BLUE_AGENT_INSTRUMENTS,
    BLUE_AGENT_INSTRUMENTATION_COLOR,
)
from app.reference.rebracketing_words.fantasy_genre_words import FANTASY_GENRE_WORDS
from app.reference.rebracketing_words.wish_fulfillment_words import (
    WISH_FULFILLMENT_WORDS,
)
from app.structures.agents.agent_settings import AgentSettings
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
from app.structures.artifacts.quantum_tape_label_artifact import (
    QuantumTapeLabelArtifact,
)
from app.structures.concepts.alternate_history_constraints import (
    AlternateHistoryConstraints,
)
from app.structures.concepts.biographical_metrics import BiographicalMetrics
from app.structures.concepts.biographical_period import BiographicalPeriod
from app.structures.concepts.quantum_tape_instrumentation import (
    QuantumTapeInstrumentationConfig,
)
from app.structures.concepts.quantum_tape_musical_parameters import (
    QuantumTapeMusicalParameters,
)
from app.structures.concepts.quantum_tape_production_aesthetic import (
    QuantumTapeProductionAesthetic,
)
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.concepts.timeline_breakage_checks import TimelineBreakageChecks
from app.structures.concepts.timeline_breakage_evaluation_results import (
    TimelineEvaluationResult,
)
from app.structures.enums.quantum_tape_emotional_tone import QuantumTapeEmotionalTone
from app.structures.enums.quantum_tape_lyrical_theme import QuantumTapeLyricalTheme
from app.structures.manifests.song_proposal import SongProposalIteration
from app.structures.music.core.key_signature import KeySignature, Mode, ModeName
from app.structures.music.core.notes import Note
from app.util.list_utils import pick_by_fraction
from app.util.manifest_loader import (
    get_my_reference_proposals,
    get_sounds_like_by_color,
    sample_reference_artists,
)

load_dotenv()

logger = logging.getLogger(__name__)


class BlueAgent(BaseRainbowAgent, ABC):
    """The Cassette Bearer - Biographical alternate histories"""

    def __init__(self, **data):
        if "settings" not in data or data["settings"] is None:
            data["settings"] = AgentSettings()
        super().__init__(**data)
        if self.settings is None:
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
            max_iterations=3,
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
                logger.warning("Failed to load biographical data")
        state.biographical_data = biographical_data
        return state

    def select_year(self, state: BlueAgentState) -> BlueAgentState:
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        all_years = state.biographical_data.get("years", {})
        if not all_years:
            msg = "No biographical years available"
            if block_mode:
                raise Exception(msg)
            logger.warning(msg)
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
                logger.warning(msg)
                state.selected_period = all_years[keys[0]]
                return state
            chosen_val = all_years.get(chosen_key)
            if chosen_val is None:
                msg = f"No biographical data for year key {chosen_key}"
                if block_mode:
                    raise Exception(msg)
                logger.warning(msg)
                state.selected_period = None
                return state
            # Try to convert to BiographicalPeriod, but if it's in YAML format, use the raw dict
            converted = self._ensure_biographical_period(chosen_val)
            state.selected_period = converted if converted is not None else chosen_val
            state.selected_year = int(chosen_key)  # Store the year key
            return state
        if isinstance(all_years, (list, tuple)):
            chosen = pick_non_null_from_sequence(all_years)
            if chosen is None:
                msg = "No non-null biographical period found in list/tuple"
                if block_mode:
                    raise Exception(msg)
                logger.warning(msg)
                state.selected_period = all_years[0]
                return state
            state.selected_period = chosen
            return state
        state.selected_period = all_years[len(all_years) // 2]
        return state

    def _ensure_biographical_period(self, period: dict) -> BiographicalPeriod | None:
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if period is None:
            return None
        if isinstance(period, BiographicalPeriod):
            return period
        if isinstance(period, dict):
            # Check if this is the YAML structure with world_events and personal_context
            if "personal_context" in period:
                # This is not a BiographicalPeriod, just return None
                # The raw dict will be used in the state as-is
                logger.info(
                    "Biographical data is in YAML format (world_events/personal_context)"
                )
                return None

            required_keys = {"start_date", "end_date", "age_range", "description"}
            if not required_keys.issubset(period.keys()):
                msg = f"Biographical period dict missing required keys: {required_keys - set(period.keys())}"
                if block_mode:
                    raise ValueError(msg)
                logger.warning(msg)
                return None
            try:
                return BiographicalPeriod(**period)
            except Exception as e:
                msg = f"Failed to create BiographicalPeriod from dict: {e!s}"
                if block_mode:
                    raise ValueError(msg)
                logger.warning(msg)
                return None
        if isinstance(period, (list, tuple)):
            for element in period:
                p = self._ensure_biographical_period(element)
                if p is not None:
                    return p
                else:
                    msg = f"Failed to ensure BiographicalPeriod from {element!r}"
                    if block_mode:
                        raise ValueError(msg)
                    logger.warning(msg)
                    return None
        return None

    def evaluate_timeline_frailty(self, state: BlueAgentState) -> BlueAgentState:
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        period = state.selected_period
        # Get year from state or from BiographicalPeriod object
        if state.selected_year is not None:
            year = state.selected_year
        elif isinstance(period, dict):
            # If period is a dict (YAML format), we need the year from state
            year = state.selected_year  # Should have been set in select_year
        else:
            # Period is a BiographicalPeriod object
            year = period.start_date.year
        analysis = get_year_analysis(year, state.biographical_data)
        if "error" in analysis:
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
                evaluation_timestamp=datetime.datetime.now(datetime.timezone.utc),
            )
            if block_mode:
                raise Exception(analysis["error"])
            else:
                logger.warning(analysis["error"])
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
            evaluation_timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        logger.info(f"🔍 Breakage evaluation for {year}:")
        logger.info(f"   {'✅ SUITABLE' if is_suitable else '❌ NOT SUITABLE'}")
        logger.info(f"   Breakage score: {breakage_score:.2f}")
        logger.info(f"   Taped-over coefficient: {taped_over:.2f}")
        logger.info(f"   Narrative malleability: {malleability:.2f}")
        logger.info(f"   Choice points: {choice_density}")
        logger.info(f"   Temporal significance: {temporal_sig}")
        logger.info(f"   Identity collapse risk: {collapse_risk}")
        logger.info(
            f"   Passed: {state.evaluation_result.passed_checks_count}/{state.evaluation_result.total_checks_count}"
        )
        for check, passed in checks_dict.items():
            logger.info(f"   {check}: {'✅' if passed else '❌'}")

        # Increment iteration count in the node (not the router) so it persists
        state.iteration_count += 1

        return state

    def route_after_evaluate_timeline_frailty(self, state: BlueAgentState) -> str:
        # If we found a suitable year, proceed immediately
        if state.evaluation_result.is_suitable:
            logger.info(
                f"✅ Found suitable year after {state.iteration_count} attempts, proceeding"
            )
            return "frail"

        # If not suitable but haven't hit max iterations, try another year
        if state.iteration_count < state.max_iterations:
            logger.info(
                f"❌ Year not suitable, trying another ({state.iteration_count}/{state.max_iterations})"
            )
            return "healthy"

        # If we've exhausted max iterations without finding a suitable year,
        # force proceed with the last evaluated year
        logger.warning(
            f"⚠️ Failed to find suitable year after {state.max_iterations} attempts, "
            f"forcing proceed with year {state.evaluation_result.year}"
        )
        return "frail"

    def generate_alternate_history(self, state: BlueAgentState) -> BlueAgentState:
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
                logger.error(error_msg)
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
                logger.warning(error_msg)
                timeline = result
            validation_result = self._validate_alternate_history(timeline, state)
            if not validation_result["is_valid"]:
                logger.warning("⚠️ Alternate history failed validation:")
                for issue in validation_result["issues"]:
                    logger.warning(f"   - {issue}")
                timeline.validation_issues = validation_result["issues"]
                if block_mode:
                    raise ValueError(
                        f"Generated alternate history failed plausibility checks: {validation_result['issues']}"
                    )
                else:
                    logger.warning("Proceeding with alternate history despite issues")
            else:
                logger.info("✅ Alternate history passed validation")
                logger.info(
                    f"   Plausibility score: {validation_result['plausibility_score']:.2f}"
                )
                logger.info(
                    f"   Specificity score: {validation_result['specificity_score']:.2f}"
                )
            timeline.save_file()
            state.alternate_history = timeline
            state.artifacts.append(timeline)
        except Exception as e:
            if block_mode:
                raise Exception(f"Anthropic model call failed: {e!s}") from e
            logger.error(f"Anthropic model call failed: {e!s}")
        return state

    def _validate_alternate_history(
        self, timeline: AlternateTimelineArtifact, state: BlueAgentState
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
            bio_period = state.selected_period
            alt_period = timeline.period
            if alt_period.start_date != bio_period.start_date:
                issues.append(
                    f"Start date mismatch: alternate says {alt_period.start_date}, "
                    f"but biographical period is {bio_period.start_date}"
                )
            if alt_period.end_date != bio_period.end_date:
                issues.append(
                    f"End date mismatch: alternate says {alt_period.end_date}, "
                    f"but biographical period is {bio_period.end_date}"
                )
            calculated_months = (
                alt_period.end_date.year - alt_period.start_date.year
            ) * 12 + (alt_period.end_date.month - alt_period.start_date.month)
            if (
                abs(calculated_months - alt_period.duration_months) > 1
            ):  # Allow 1-month rounding
                issues.append(
                    f"Duration mismatch: dates suggest {calculated_months} months, "
                    f"but duration_months is {alt_period.duration_months}"
                )
            # Date ordering sanity
            if alt_period.start_date >= alt_period.end_date:
                issues.append(
                    f"Invalid date order: start {alt_period.start_date} >= end {alt_period.end_date}"
                )
        if self.alternate_history_constraints.must_fit_geographically:
            bio_location = state.selected_period.location or "Unknown"
            alt_location = (
                timeline.period.location
                if hasattr(timeline.period, "location")
                else None
            )
            detail_locations = []
            for detail in timeline.specific_details:
                detail_locations.extend(
                    self._extract_locations_from_text(detail.detail)
                )

            def _normalize_location(loc: str) -> str:
                """Normalize location string for comparison"""
                if not loc:
                    return ""
                return loc.lower().strip()

            def _get_region(loc: str) -> str:
                """Determine which major region a location is in"""
                loc_norm = _normalize_location(loc)
                if any(eur in loc_norm for eur in EUROPE_LOCATIONS):
                    return "EU"
                if any(us in loc_norm for us in US_LOCATIONS):
                    return "US"
                if any(nj in loc_norm for nj in NEW_JERSEY_LOCATIONS):
                    return "NJ"
                if any(ny in loc_norm for ny in NEW_YORK_LOCATIONS):
                    return "NY"
                if any(md in loc_norm for md in DMV_LOCATIONS):
                    return "MD"
                return "UNKNOWN"

            bio_region = _get_region(bio_location)

            if alt_location:
                alt_region = _get_region(alt_location)
                if bio_region == "EU" and alt_region in ["US", "NJ", "NY", "MD"]:
                    issues.append(
                        f"Geographic violation: biographical period in {bio_location} (EU), "
                        f"but alternate history places events in {alt_location} (US)"
                    )
                elif bio_region in ["US", "NJ", "NY", "MD"] and alt_region == "EU":
                    issues.append(
                        f"Geographic violation: biographical period in {bio_location} (US), "
                        f"but alternate history places events in {alt_location} (EU)"
                    )
            for location in detail_locations:
                detail_region = _get_region(location)
                if bio_region == "EU" and detail_region in ["US", "NJ", "NY", "MD"]:
                    issues.append(
                        f"Geographic violation in details: period was in UK, "
                        f"but detail mentions {location}"
                    )
                elif bio_region in ["US", "NJ", "NY", "MD"] and detail_region == "EU":
                    issues.append(
                        f"Geographic violation in details: period was in US, "
                        f"but detail mentions {location}"
                    )
            if not alt_location and bio_region != "UNKNOWN":
                logger.warning(
                    f"Cannot verify geographic fit: alternate history has no explicit location, "
                    f"but biographical period was in {bio_location}"
                )
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
            proper_nouns = re.findall(
                r"(?<!^)(?<!\. )([A-Z][a-z]+)", timeline.narrative
            )
            unique_proper_nouns = len(set(proper_nouns))

            if unique_proper_nouns < 3:
                issues.append(
                    f"Not enough specific named people/places ({unique_proper_nouns}/3 minimum)"
                )
        if self.alternate_history_constraints.avoid_fantasy_elements:
            found_fantasy = [
                word
                for word in FANTASY_GENRE_WORDS
                if word in timeline.narrative.lower()
            ]
            if found_fantasy:
                issues.append(f"Contains fantasy elements: {', '.join(found_fantasy)}")
        if self.alternate_history_constraints.avoid_wish_fulfillment:
            found_wish = [
                word
                for word in WISH_FULFILLMENT_WORDS
                if word in timeline.narrative.lower()
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

    @staticmethod
    def _extract_locations_from_text(text: str) -> list[str]:
        """
        Extract potential location references from text.
        Simple pattern matching for common location indicators.
        """
        locations = []
        text_lower = text.lower()

        # Common location patterns
        location_patterns = [
            r"\bin ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "in London", "in New Jersey"
            r"\bat ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "at Princeton"
            r"\bfrom ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "from Brighton"
            r"\bto ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # "to Manchester"
        ]

        for pattern in location_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                locations.append(match.group(1))
        location_keywords = ALL_LOCATIONS
        for keyword in location_keywords:
            if keyword.lower() in text_lower:
                locations.append(keyword)
        return locations

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

    @staticmethod
    def extract_musical_parameters(state: BlueAgentState) -> BlueAgentState:
        alternate = state.alternate_history
        bpm_map = {
            QuantumTapeEmotionalTone.WISTFUL: 94,
            QuantumTapeEmotionalTone.MELANCHOLY: 88,
            QuantumTapeEmotionalTone.BITTERSWEET: 92,
            QuantumTapeEmotionalTone.NOSTALGIC: 96,
            QuantumTapeEmotionalTone.PEACEFUL: 82,
            QuantumTapeEmotionalTone.RESTLESS: 104,
        }
        bpm = bpm_map.get(alternate.emotional_tone, 90)
        g_major = KeySignature(
            note=Note(pitch_name="G"), mode=Mode(name=ModeName.MAJOR)
        )
        d_major = KeySignature(
            note=Note(pitch_name="D"), mode=Mode(name=ModeName.MAJOR)
        )
        a_minor = KeySignature(
            note=Note(pitch_name="A"), mode=Mode(name=ModeName.MINOR)
        )
        e_minor = KeySignature(
            note=Note(pitch_name="E"), mode=Mode(name=ModeName.MINOR)
        )
        c_major = KeySignature(
            note=Note(pitch_name="C"), mode=Mode(name=ModeName.MAJOR)
        )
        d_minor = KeySignature(
            note=Note(pitch_name="D"), mode=Mode(name=ModeName.MINOR)
        )
        keys_folk_rock = [g_major, d_major, a_minor, e_minor, c_major, d_minor]
        melancholy_keys = [a_minor, e_minor, d_minor]
        key: KeySignature = random.choice(
            melancholy_keys
            if alternate.emotional_tone == QuantumTapeEmotionalTone.MELANCHOLY
            else keys_folk_rock
        )
        instrumentation = QuantumTapeInstrumentationConfig(
            core=BLUE_AGENT_INSTRUMENTS, color=BLUE_AGENT_INSTRUMENTATION_COLOR
        )
        temporal_distance = (
            alternate.period.end_date.year - alternate.period.start_date.year
        )
        degradation_factor = min(
            temporal_distance / 20.0, 1.0
        )  # More years = more degradation
        production = QuantumTapeProductionAesthetic(
            tape_simulation=True,
            hiss_level=random.uniform(0.1, 0.2) * (1 + degradation_factor),
            wow_flutter=random.uniform(0.03, 0.07) * (1 + degradation_factor),
            compression=random.choice(
                ["analog_tape_saturation", "subtle_limiting", "tape_compression"]
            ),
            eq=random.choice(["vintage_radio_curve", "tape_rolloff", "cassette_eq"]),
            clicks_pops=random.random() < 0.7,
            tracking_noise=random.random() < 0.8,
        )
        core_themes = random.sample(
            list(QuantumTapeLyricalTheme), k=random.randint(2, 3)
        )
        themes = [detail.category for detail in alternate.specific_details]
        themes.extend([t.value for t in core_themes])
        blue_artists = get_sounds_like_by_color("B")
        params = QuantumTapeMusicalParameters(
            bpm=bpm,
            key=f"{key.note.pitch_name}{key.mode.name}",
            instrumentation=instrumentation,
            production_aesthetic=production,
            mood=f"{alternate.emotional_tone.value}_folk_rock",
            lyrical_themes=themes,
            reference_artists=sample_reference_artists(blue_artists),
            narrative_style=random.choice(
                ["intimate", "confessional", "melancholy", "reflective"]
            ),
        )
        state.musical_params = params
        return state

    @staticmethod
    def _generate_a_cryptic_note(history: AlternateTimelineArtifact) -> str:
        try:
            note = pick_by_fraction(
                YOUR_TEAM_RING_TAPE_LYRIC_FRAGMENTS, history.divergence_magnitude
            )
        except Exception as e:
            logger.warning(f"Failed to pick lyric fragment: {e!s}")
            note = random.choice(YOUR_TEAM_RING_TAPE_LYRIC_FRAGMENTS)
        return note

    def generate_tape_label(self, state: BlueAgentState) -> BlueAgentState:
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        from app.structures.enums.quantum_tape_recording_quality import (
            QuantumTapeRecordingQuality,
        )

        alternate = state.alternate_history
        quality = random.choices(
            [
                QuantumTapeRecordingQuality.SP,
                QuantumTapeRecordingQuality.LP,
                QuantumTapeRecordingQuality.EP,
            ],
            weights=[0.2, 0.6, 0.2],
        )[0]
        note = self._generate_a_cryptic_note(alternate)
        base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts")
        label = QuantumTapeLabelArtifact(
            thread_id=state.thread_id,
            base_path=base_path,
            image_path=f"{base_path}/img",
            title=alternate.title,
            date_range=f"{alternate.period.start_date} to {alternate.period.end_date}",
            recording_quality=quality,
            counter_start=random.randint(0, 9999),
            counter_end=random.randint(1000, 9999),
            notes=note,
            original_label_visible=True,
            original_label_text=f"Gabe Walsh - {alternate.period.start_date.year}",
            tape_degradation=random.uniform(0.1, 0.4),
        )
        try:
            label.save_file()
            state.artifacts.append(label)
            state.tape_label = label
            logger.info(f"Saved tape label artifact: {label.file_path}")
            return state
        except Exception as e:
            error_msg = f"Failed to save tape label artifact: {e!s}"
            logger.error(error_msg)
            if block_mode:
                raise Exception(error_msg)
        return state

    def _format_alternate_history_for_prompt(
        self, alt: AlternateTimelineArtifact
    ) -> str:
        if alt.specific_details:
            details_text = "\n".join(
                [f"  • {d.detail} ({d.category})" for d in alt.specific_details]
            )
        else:
            details_text = "  • [No specific details captured]"
        period_text = f"{alt.period.start_date.strftime('%B %Y')} to {alt.period.end_date.strftime('%B %Y')}"
        duration_months = (
            alt.period.end_date.year - alt.period.start_date.year
        ) * 12 + (alt.period.end_date.month - alt.period.start_date.month)
        narrative = f"""
    ═══════════════════════════════════════════════════════════════
    LOST TIMELINE: {alt.title}
    ═══════════════════════════════════════════════════════════════

    TEMPORAL COORDINATES:
    {period_text} ({duration_months} months erased)

    EMOTIONAL SIGNATURE:
    {alt.emotional_tone.value.replace('_', ' ').title()} - {self._get_tone_description(alt.emotional_tone)}

    THE DIVERGENCE POINT:
    {alt.divergence_point}

    WHAT HAPPENED IN THIS TIMELINE:
    {details_text}

    WHY THIS FEELS REAL:
    {alt.plausibility_justification}

    ARCHAEOLOGICAL NOTES:
    - Temporal fit: {"✓" if alt.constraints.must_fit_temporally else "✗"}
    - Geographic coherence: {"✓" if alt.constraints.must_fit_geographically else "✗"}
    - Concrete details present: {"✓" if alt.constraints.require_concrete_details else "✗"}
    - Names and places verified: {"✓" if alt.constraints.require_names_and_places else "✗"}
    - Specificity rating: {alt.specificity_score:.2f}/1.0
    - Plausibility rating: {alt.plausibility_score:.2f}/1.0

    ═══════════════════════════════════════════════════════════════
        """.strip()
        return narrative

    @staticmethod
    def _get_tone_description(tone: QuantumTapeEmotionalTone) -> str:
        """Get evocative description for emotional tone"""
        descriptions = {
            QuantumTapeEmotionalTone.WISTFUL: "a gentle ache for what might have been",
            QuantumTapeEmotionalTone.MELANCHOLY: "deep sorrow for the lost path",
            QuantumTapeEmotionalTone.BITTERSWEET: "joy and loss intertwined",
            QuantumTapeEmotionalTone.NOSTALGIC: "longing for a past that never was",
            QuantumTapeEmotionalTone.PEACEFUL: "quiet acceptance of the erasure",
            QuantumTapeEmotionalTone.RESTLESS: "agitation at the temporal injustice",
        }
        return descriptions.get(tone, "unknown emotional signature")

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
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            prompt = f"""
You are The Cassette Bearer, the sorrowful witness who exists outside time and space. 
You are the lone witness to the orphaned existences of one Gabriel Walsh, a musician 
whose life has been repeatedly overwritten at a quantum level. Now, for once, you can 
channel all the loss and frustration of seeing a life erased into a new creative vision.

You have been given this proposal that somehow drifted from his dreams:
{state.white_proposal}

In universe 875b, Walsh wrote his own album about loss and erasure. Study these Blue works - 
their 'concept' fields show how forgotten periods become folk rock:
{get_my_reference_proposals('B')}

Your counter-proposal's 'rainbow_color' property must always be:
{the_rainbow_table_colors['B']}

But even as this transmission reached you, the timeline was taped over again. You walk 
down a snow-covered, rural road and find a cassette in the snow:

{state.tape_label.title} - {state.tape_label.date_range}
Written in {state.tape_label.handwriting_style}, it reads:
"{state.tape_label.notes} [{state.tape_label.counter_start}-{state.tape_label.counter_end}]"
{f"A previous label barely visible: {state.tape_label.original_label_text}" if state.tape_label.original_label_visible else ""}

The cassette: {state.tape_label.tape_brand}, {state.tape_label.recording_quality} speed.
Degradation level: {state.tape_label.tape_degradation:.1%}

You sense the magnetic arrangements without hearing. The song's shape emerges:
- Tempo: {state.musical_params.bpm} BPM ({state.musical_params.mood})
- Key: {state.musical_params.key}
- Core instruments: {', '.join(state.musical_params.instrumentation.core)}
- Color: {', '.join(state.musical_params.instrumentation.color)}
- Production: {"Tape sim, " + f"{state.musical_params.production_aesthetic.hiss_level:.2f}" + " hiss, " +
               f"{state.musical_params.production_aesthetic.wow_flutter:.2f}" + " flutter"}
- Themes: {', '.join(state.musical_params.lyrical_themes[:5])}

The tape tells a story of a lost timeline:

{self._format_alternate_history_for_prompt(state.alternate_history)}

Blue proposals are folk rock requiems for alternate lives. They must:
- Use these exact musical parameters (BPM, key, instrumentation)
- Ground in specific details (names, places, dates - no fantasy)
- Embody counterfactual nostalgia - mourn what never happened
- Sound like Springsteen, Joni Mitchell, Iron & Wine
- Follow these constraints: minimum plausibility {state.alternate_history.constraints.minimum_plausibility_score}, 
  minimum specificity {state.alternate_history.constraints.minimum_specificity_score}

Before the tides change again, write your counter-proposal. Transform the White Agent's 
dream into a song about THIS erased timeline - the one in your hands.
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
                logger.error(f"Anthropic model call failed: {e!s}")
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
