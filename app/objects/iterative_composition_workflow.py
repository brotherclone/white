import uuid

from typing import Dict, Any

from app.enums.workflow_stage import WorkflowStage
from app.enums.chord_quality import ChordQuality
from app.objects.workflow_state import WorkflowState
from app.objects.iteration_feedback import IterationFeedback
from app.objects.chord_progression import ChordProgression
from app.objects.chord import Chord

class IterativeCompositionWorkflow:
    """Manages iterative collaboration between agents"""

    def __init__(self, andy_coordinator):
        self.andy = andy_coordinator
        self.state = None
        self.quality_thresholds = {
            WorkflowStage.PLANNING: 0.7,
            WorkflowStage.CHORD_GENERATION: 0.6,
            WorkflowStage.LYRICS_GENERATION: 0.8,
            WorkflowStage.MELODY_GENERATION: 0.7,
            WorkflowStage.ARRANGEMENT: 0.6,
        }

    def start_composition(self,
                          rainbow_color=None,
                          user_preferences: Dict = None,
                          max_iterations: int = 5) -> str:
        """Start a new iterative composition session"""

        session_id = str(uuid.uuid4())
        self.state = WorkflowState(
            session_id=session_id,
            current_stage=WorkflowStage.PLANNING,
            current_iteration=0,
            max_iterations=max_iterations
        )

        print(f"🎵 Starting composition session: {session_id}")
        return session_id

    def execute_workflow(self, rainbow_color=None) -> Dict[str, Any]:
        """Execute the complete iterative workflow"""

        if not self.state:
            self.start_composition()

        # Stage 1: Planning
        self._execute_planning_stage(rainbow_color)

        # Stage 2: Chord Generation
        self._execute_chord_generation_stage()

        # Stage 3: Lyrics Generation (if needed)
        if self.state.plan and self.state.plan.get("vocals", True):
            self._execute_lyrics_stage()

        # Stage 4: Melody Generation
        self._execute_melody_stage()

        # Stage 5: Arrangement
        self._execute_arrangement_stage()

        # Stage 6: Final validation
        self._execute_validation_stage()

        return self._compile_final_output()

    def _execute_planning_stage(self, rainbow_color=None):
        """Execute planning stage with iteration"""
        print(f"\n🎯 Stage: {WorkflowStage.PLANNING.value.upper()}")

        self.state.current_stage = WorkflowStage.PLANNING

        for iteration in range(self.state.max_iterations):
            self.state.current_iteration = iteration
            print(f"  Iteration {iteration + 1}")

            # Generate plan
            if iteration == 0:
                # plan = self.andy.plan_agent._generate_plan(rainbow_color=rainbow_color)
                # ^ this is colliding with some previous ideas
                plan = self.andy.plan_agent._generate_plan()

            else:
                # Refine based on feedback
                previous_feedback = self.state.get_latest_feedback(WorkflowStage.PLANNING)
                plan = self._refine_plan(plan, previous_feedback)

            # Evaluate plan
            feedback = self._evaluate_plan(plan, iteration)
            self.state.feedback_history.append(feedback)

            if feedback.approval:
                self.state.plan = plan
                print(f"  ✅ Plan approved after {iteration + 1} iterations")
                break
            else:
                print(f"  🔄 Plan needs refinement: {feedback.feedback}")

        if not self.state.plan:
            print("  ⚠️ Using best plan after max iterations")
            self.state.plan = plan

    def _execute_chord_generation_stage(self):
        """Execute chord generation with iteration"""
        print(f"\n🎼 Stage: {WorkflowStage.CHORD_GENERATION.value.upper()}")

        self.state.current_stage = WorkflowStage.CHORD_GENERATION

        for iteration in range(self.state.max_iterations):
            self.state.current_iteration = iteration
            print(f"  Iteration {iteration + 1}")

            # Generate chord progressions
            if hasattr(self.andy, 'chord_generator'):
                song_plan = self.andy.plan_agent.env.plan_to_rainbow_song_plan(self.state.plan)

                chord_progressions = []
                previous_progression = None

                for section in song_plan.structure:
                    progression = self.andy.chord_generator.generate_progression_for_section(
                        section=section,
                        key=song_plan.key,
                        mood_tags=song_plan.moods or [],
                        previous_progression=previous_progression
                    )
                    chord_progressions.append(progression)
                    previous_progression = progression
            else:
                # Fallback to simple chord generation
                chord_progressions = self._simple_chord_generation(self.state.plan)

            # Evaluate chord progressions
            feedback = self._evaluate_chord_progressions(chord_progressions, iteration)
            self.state.feedback_history.append(feedback)

            if feedback.approval:
                self.state.chord_progressions = [prog.dict() for prog in chord_progressions]
                print(f"  ✅ Chord progressions approved after {iteration + 1} iterations")
                break
            else:
                print(f"  🔄 Chord progressions need refinement: {feedback.feedback}")

        if not self.state.chord_progressions:
            print("  ⚠️ Using best chord progressions after max iterations")
            self.state.chord_progressions = [prog.dict() for prog in chord_progressions]

    def _execute_lyrics_stage(self):
        """Execute lyrics generation with iteration"""
        print(f"\n🎤 Stage: {WorkflowStage.LYRICS_GENERATION.value.upper()}")

        self.state.current_stage = WorkflowStage.LYRICS_GENERATION

        for iteration in range(self.state.max_iterations):
            self.state.current_iteration = iteration
            print(f"  Iteration {iteration + 1}")

            # Generate lyrics using Dorthy
            lyrics_data = self._generate_lyrics_with_dorthy(iteration)

            # Evaluate lyrics
            feedback = self._evaluate_lyrics(lyrics_data, iteration)
            self.state.feedback_history.append(feedback)

            if feedback.approval:
                self.state.lyrics_data = lyrics_data
                print(f"  ✅ Lyrics approved after {iteration + 1} iterations")
                break
            else:
                print(f"  🔄 Lyrics need refinement: {feedback.feedback}")

        if not self.state.lyrics_data:
            print("  ⚠️ Using best lyrics after max iterations")
            self.state.lyrics_data = lyrics_data

    def _execute_melody_stage(self):
        """Execute melody generation with iteration"""
        print(f"\n🎵 Stage: {WorkflowStage.MELODY_GENERATION.value.upper()}")

        self.state.current_stage = WorkflowStage.MELODY_GENERATION

        for iteration in range(self.state.max_iterations):
            self.state.current_iteration = iteration
            print(f"  Iteration {iteration + 1}")

            # Generate melody using Nancarrow
            melody_data = self._generate_melody_with_nancarrow(iteration)

            # Evaluate melody
            feedback = self._evaluate_melody(melody_data, iteration)
            self.state.feedback_history.append(feedback)

            if feedback.approval:
                self.state.melody_data = melody_data
                print(f"  ✅ Melody approved after {iteration + 1} iterations")
                break
            else:
                print(f"  🔄 Melody needs refinement: {feedback.feedback}")

        if not self.state.melody_data:
            print("  ⚠️ Using best melody after max iterations")
            self.state.melody_data = melody_data

    def _execute_arrangement_stage(self):
        """Execute arrangement stage with iteration"""
        print(f"\n🎛️ Stage: {WorkflowStage.ARRANGEMENT.value.upper()}")

        self.state.current_stage = WorkflowStage.ARRANGEMENT

        for iteration in range(self.state.max_iterations):
            self.state.current_iteration = iteration
            print(f"  Iteration {iteration + 1}")

            # Generate arrangement using Martin
            arrangement_data = self._generate_arrangement_with_martin(iteration)

            # Evaluate arrangement
            feedback = self._evaluate_arrangement(arrangement_data, iteration)
            self.state.feedback_history.append(feedback)

            if feedback.approval:
                self.state.arrangement_data = arrangement_data
                print(f"  ✅ Arrangement approved after {iteration + 1} iterations")
                break
            else:
                print(f"  🔄 Arrangement needs refinement: {feedback.feedback}")

        if not self.state.arrangement_data:
            print("  ⚠️ Using best arrangement after max iterations")
            self.state.arrangement_data = arrangement_data

    def _execute_validation_stage(self):
        """Final validation and quality check"""
        print(f"\n✅ Stage: {WorkflowStage.VALIDATION.value.upper()}")

        self.state.current_stage = WorkflowStage.VALIDATION

        # Overall quality assessment
        overall_quality = self._assess_overall_quality()

        feedback = IterationFeedback(
            agent_name="validator",
            stage=WorkflowStage.VALIDATION,
            iteration=0,
            feedback=f"Overall composition quality: {overall_quality:.2f}",
            suggested_changes={},
            confidence_score=overall_quality,
            approval=overall_quality >= 0.7
        )

        self.state.feedback_history.append(feedback)

        if feedback.approval:
            print(f"  ✅ Composition validated with quality score: {overall_quality:.2f}")
        else:
            print(f"  ⚠️ Composition quality below threshold: {overall_quality:.2f}")

    # Evaluation methods
    def _evaluate_plan(self, plan: Dict, iteration: int) -> IterationFeedback:
        """Evaluate the generated plan"""

        # Check for required elements
        required_keys = ["key", "tempo", "sections"]
        missing_keys = [k for k in required_keys if k not in plan]

        if missing_keys:
            return IterationFeedback(
                agent_name="subutai",
                stage=WorkflowStage.PLANNING,
                iteration=iteration,
                feedback=f"Missing required elements: {missing_keys}",
                suggested_changes={"add_keys": missing_keys},
                confidence_score=0.3,
                approval=False
            )

        # Check tempo range
        tempo = plan.get("tempo", 120)
        if tempo < 60 or tempo > 200:
            return IterationFeedback(
                agent_name="subutai",
                stage=WorkflowStage.PLANNING,
                iteration=iteration,
                feedback=f"Tempo {tempo} is outside reasonable range",
                suggested_changes={"tempo": max(60, min(200, tempo))},
                confidence_score=0.4,
                approval=False
            )

        # Assess overall coherence
        coherence_score = 0.8 + (iteration * -0.1)  # Slight degradation with iterations
        threshold = self.quality_thresholds[WorkflowStage.PLANNING]

        return IterationFeedback(
            agent_name="subutai",
            stage=WorkflowStage.PLANNING,
            iteration=iteration,
            feedback="Plan structure looks good" if coherence_score >= threshold else "Plan needs more coherence",
            suggested_changes={},
            confidence_score=coherence_score,
            approval=coherence_score >= threshold
        )

    def _evaluate_chord_progressions(self, progressions: list, iteration: int) -> IterationFeedback:
        """Evaluate chord progressions"""

        if not progressions:
            return IterationFeedback(
                agent_name="chord_generator",
                stage=WorkflowStage.CHORD_GENERATION,
                iteration=iteration,
                feedback="No chord progressions generated",
                suggested_changes={},
                confidence_score=0.0,
                approval=False
            )

        # Check for variety and coherence
        total_chords = sum(len(prog.chords) for prog in progressions)
        unique_chords = len(set(str(chord) for prog in progressions for chord in prog.chords))

        variety_score = min(1.0, unique_chords / max(1, total_chords * 0.5))

        # Check key consistency
        keys = [prog.key for prog in progressions]
        key_consistency = len(set(keys)) <= 2  # Allow for one modulation

        overall_score = (variety_score + (0.8 if key_consistency else 0.3)) / 2
        threshold = self.quality_thresholds[WorkflowStage.CHORD_GENERATION]

        feedback_msg = "Chord progressions show good variety and consistency"
        if not key_consistency:
            feedback_msg = "Too many key changes - maintain more consistency"
        elif variety_score < 0.4:
            feedback_msg = "Chord progressions lack variety - add more diverse chords"

        return IterationFeedback(
            agent_name="chord_generator",
            stage=WorkflowStage.CHORD_GENERATION,
            iteration=iteration,
            feedback=feedback_msg,
            suggested_changes={"variety_score": variety_score, "key_consistency": key_consistency},
            confidence_score=overall_score,
            approval=overall_score >= threshold
        )

    def _evaluate_lyrics(self, lyrics_data: Dict, iteration: int) -> IterationFeedback:
        """Evaluate generated lyrics"""

        if not lyrics_data or not lyrics_data.get("sections"):
            return IterationFeedback(
                agent_name="dorthy",
                stage=WorkflowStage.LYRICS_GENERATION,
                iteration=iteration,
                feedback="No lyrics generated",
                suggested_changes={},
                confidence_score=0.0,
                approval=False
            )

        # Check for content in vocal sections
        vocal_sections = ["verse", "chorus", "bridge"]
        lyrics_sections = lyrics_data["sections"]

        has_content = any(
            section.get("lyrics") and len(section["lyrics"].strip()) > 10
            for section in lyrics_sections
            if any(vs in section.get("name", "").lower() for vs in vocal_sections)
        )

        if not has_content:
            return IterationFeedback(
                agent_name="dorthy",
                stage=WorkflowStage.LYRICS_GENERATION,
                iteration=iteration,
                feedback="Vocal sections need more lyrical content",
                suggested_changes={"add_lyrics_to": vocal_sections},
                confidence_score=0.3,
                approval=False
            )

        # Check for mood consistency
        plan_moods = self.state.plan.get("moods", [])
        mood_consistency_score = 0.8  # Simplified evaluation

        threshold = self.quality_thresholds[WorkflowStage.LYRICS_GENERATION]

        return IterationFeedback(
            agent_name="dorthy",
            stage=WorkflowStage.LYRICS_GENERATION,
            iteration=iteration,
            feedback="Lyrics show good content and mood consistency" if mood_consistency_score >= threshold else "Lyrics need better mood alignment",
            suggested_changes={},
            confidence_score=mood_consistency_score,
            approval=mood_consistency_score >= threshold
        )

    def _evaluate_melody(self, melody_data: Dict, iteration: int) -> IterationFeedback:
        """Evaluate generated melody"""

        if not melody_data:
            return IterationFeedback(
                agent_name="nancarrow",
                stage=WorkflowStage.MELODY_GENERATION,
                iteration=iteration,
                feedback="No melody data generated",
                suggested_changes={},
                confidence_score=0.0,
                approval=False
            )

        # Check for melodic content
        has_notes = melody_data.get("note_count", 0) > 0
        melodic_range = melody_data.get("range_semitones", 0)

        range_score = min(1.0, melodic_range / 12.0)  # Good range is about an octave

        overall_score = 0.7 if has_notes else 0.2
        if has_notes:
            overall_score = (overall_score + range_score) / 2

        threshold = self.quality_thresholds[WorkflowStage.MELODY_GENERATION]

        feedback_msg = "Melody shows good range and content"
        if not has_notes:
            feedback_msg = "No melodic content generated"
        elif range_score < 0.3:
            feedback_msg = "Melody needs wider range"

        return IterationFeedback(
            agent_name="nancarrow",
            stage=WorkflowStage.MELODY_GENERATION,
            iteration=iteration,
            feedback=feedback_msg,
            suggested_changes={"range_score": range_score},
            confidence_score=overall_score,
            approval=overall_score >= threshold
        )

    def _evaluate_arrangement(self, arrangement_data: Dict, iteration: int) -> IterationFeedback:
        """Evaluate arrangement"""

        if not arrangement_data:
            return IterationFeedback(
                agent_name="martin",
                stage=WorkflowStage.ARRANGEMENT,
                iteration=iteration,
                feedback="No arrangement data generated",
                suggested_changes={},
                confidence_score=0.0,
                approval=False
            )

        # Check for instrumentation variety
        instruments = arrangement_data.get("instruments", [])
        instrument_count = len(instruments)

        # Check for dynamic variation
        has_dynamics = arrangement_data.get("dynamic_variation", False)

        variety_score = min(1.0, instrument_count / 5.0)  # Good arrangement has ~5 instruments
        dynamics_score = 0.8 if has_dynamics else 0.4

        overall_score = (variety_score + dynamics_score) / 2
        threshold = self.quality_thresholds[WorkflowStage.ARRANGEMENT]

        return IterationFeedback(
            agent_name="martin",
            stage=WorkflowStage.ARRANGEMENT,
            iteration=iteration,
            feedback="Arrangement shows good variety and dynamics" if overall_score >= threshold else "Arrangement needs more instrumentation or dynamics",
            suggested_changes={"instrument_count": instrument_count, "has_dynamics": has_dynamics},
            confidence_score=overall_score,
            approval=overall_score >= threshold
        )

    # Generation methods for agents that aren't fully implemented yet
    @staticmethod
    def _simple_chord_generation(plan: Dict) -> list:
        """Fallback chord generation"""

        # Simple I-V-vi-IV progression
        chords = [
            Chord(root="C", quality=ChordQuality.MAJOR),
            Chord(root="G", quality=ChordQuality.MAJOR),
            Chord(root="A", quality=ChordQuality.MINOR),
            Chord(root="F", quality=ChordQuality.MAJOR),
        ]

        progression = ChordProgression(
            chords=chords,
            section_name="Generated Section",
            bars_per_chord=[1, 1, 1, 1],
            key=plan.get("key", "C major")
        )

        return [progression]

    def _generate_lyrics_with_dorthy(self, iteration: int) -> Dict:
        """Generate lyrics using Dorthy agent"""

        # Placeholder - implement actual Dorthy processing
        if hasattr(self.andy.lyrics_agent, 'vector_store') and self.andy.lyrics_agent.vector_store:
            # Use vector store for lyrics generation
            mood_query = " ".join(self.state.plan.get("moods", ["mysterious"]))

            # This would be actual vector search in real implementation
            similar_lyrics = ["sample lyric line 1", "sample lyric line 2"]

            lyrics_data = {
                "sections": [
                    {
                        "name": "verse_1",
                        "lyrics": "In the shadows of the night\nWhere mysteries unfold\nSecrets whispered in the dark\nStories left untold"
                    },
                    {
                        "name": "chorus",
                        "lyrics": "Through the veil of time\nWe search for what is true\nIn the depths of the unknown\nWe find ourselves anew"
                    }
                ],
                "mood_alignment": 0.8,
                "iteration": iteration
            }
        else:
            # Simple fallback
            lyrics_data = {
                "sections": [
                    {"name": "verse_1", "lyrics": "Generated verse lyrics"},
                    {"name": "chorus", "lyrics": "Generated chorus lyrics"}
                ],
                "mood_alignment": 0.6,
                "iteration": iteration
            }

        return lyrics_data

    def _generate_melody_with_nancarrow(self, iteration: int) -> Dict:
        """Generate melody using Nancarrow agent"""

        # Placeholder melody generation
        melody_data = {
            "note_count": 32,
            "range_semitones": 12,
            "key": self.state.plan.get("key", "C major"),
            "sections": [
                {
                    "name": "verse_melody",
                    "notes": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
                    "durations": [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0]
                },
                {
                    "name": "chorus_melody",
                    "notes": ["G4", "A4", "B4", "C5", "B4", "A4", "G4", "F4"],
                    "durations": [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0]
                }
            ],
            "iteration": iteration
        }

        return melody_data

    def _generate_arrangement_with_martin(self, iteration: int) -> Dict:
        """Generate arrangement using Martin agent"""

        # Placeholder arrangement
        arrangement_data = {
            "instruments": ["piano", "bass", "drums", "synth", "guitar"],
            "dynamic_variation": True,
            "sections": [
                {
                    "name": "verse",
                    "instruments": ["piano", "bass", "light_drums"],
                    "dynamics": "soft"
                },
                {
                    "name": "chorus",
                    "instruments": ["piano", "bass", "drums", "synth", "guitar"],
                    "dynamics": "full"
                }
            ],
            "iteration": iteration
        }

        return arrangement_data

    # Refinement methods
    def _refine_plan(self, current_plan: Dict, feedback: IterationFeedback) -> Dict:
        """Refine plan based on feedback"""

        refined_plan = current_plan.copy()

        if feedback and feedback.suggested_changes:
            changes = feedback.suggested_changes

            # Apply suggested tempo changes
            if "tempo" in changes:
                refined_plan["tempo"] = changes["tempo"]

            # Add missing keys
            if "add_keys" in changes:
                for key in changes["add_keys"]:
                    if key == "sections":
                        refined_plan["sections"] = refined_plan.get("sections", 4)
                    elif key not in refined_plan:
                        refined_plan[key] = self._get_default_value(key)

        return refined_plan

    def _get_default_value(self, key: str) -> Any:
        """Get default values for missing plan elements"""
        defaults = {
            "key": "C major",
            "tempo": 120,
            "sections": 4,
            "motifs": [0.5, 0.5, 0.5],
            "instruments": [1, 2, 3, 4, 5]
        }
        return defaults.get(key, None)

    def _assess_overall_quality(self) -> float:
        """Assess overall quality of the composition"""

        stage_scores = {}

        for feedback in self.state.feedback_history:
            if feedback.approval and feedback.stage not in stage_scores:
                stage_scores[feedback.stage] = feedback.confidence_score

        if not stage_scores:
            return 0.0

        # Weight different stages
        weights = {
            WorkflowStage.PLANNING: 0.2,
            WorkflowStage.CHORD_GENERATION: 0.25,
            WorkflowStage.LYRICS_GENERATION: 0.2,
            WorkflowStage.MELODY_GENERATION: 0.25,
            WorkflowStage.ARRANGEMENT: 0.1
        }

        weighted_score = 0.0
        total_weight = 0.0

        for stage, score in stage_scores.items():
            weight = weights.get(stage, 0.1)
            weighted_score += score * weight
            total_weight += weight

        return weighted_score / max(total_weight, 0.1)

    def _compile_final_output(self) -> Dict[str, Any]:
        """Compile all outputs into final composition"""

        output = {
            "session_id": self.state.session_id,
            "plan": self.state.plan,
            "chord_progressions": self.state.chord_progressions,
            "lyrics": self.state.lyrics_data,
            "melody": self.state.melody_data,
            "arrangement": self.state.arrangement_data,
            "quality_score": self._assess_overall_quality(),
            "iterations_used": {
                stage.value: len([f for f in self.state.feedback_history if f.stage == stage])
                for stage in WorkflowStage
            },
            "feedback_summary": [
                {
                    "stage": f.stage.value,
                    "final_approval": f.approval,
                    "confidence": f.confidence_score,
                    "feedback": f.feedback
                }
                for f in self.state.feedback_history
                if f.approval  # Only include approved iterations
            ]
        }

        return output
