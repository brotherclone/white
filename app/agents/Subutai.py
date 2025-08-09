import os
import random
import yaml
import uuid
import logging
import shutil
import glob
from datetime import datetime
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from typing import Any, Optional, List, Dict, Union
from pydantic import PrivateAttr

from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.objects.db_models.concept_schema import ConceptSchema
from app.resources.prompts.concepts import preambles
from app.utils.db_util import create_concept, get_random_concept
from app.objects.song_plan import RainbowSongPlanStarter, RainbowSongPlan
from app.utils.string_util import quote_yaml_values
from app.enums.plan_state import PlanState
from app.objects.plan_feedback import RainbowPlanFeedback

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv(verbose=True)


class Subutai(BaseRainbowAgent):
    """
    Subutai agent is responsible for generating, critiquing, and improving song plans.

    This agent uses AI to generate creative concepts for songs, create structured plans,
    and improve them based on human feedback. It manages the lifecycle of plans from
    initial concept to refined implementation.

    The agent works with three key components:
    1. Manifests - Raw song data in YAML format
    2. Reference Plans - Reviewed and approved plans that serve as examples
    3. Generated Plans - New plans created by the agent

    The workflow involves:
    - Learning from reference plans
    - Generating new plans based on learned patterns
    - Connecting plans to manifests
    - Creating reference plans from successful manifests
    """
    generator: Any = None
    model: Optional[Any] = None
    tokenizer: Any = None
    plan_data: Any = None
    vector_store: Any = None
    preambles: list[str] = []
    _starter_directory: str = PrivateAttr()
    _unreviewed_directory: str = PrivateAttr()
    _reviewed_directory: str = PrivateAttr()
    _reference_directory: str = PrivateAttr()
    _reference_unreviewed_directory: str = PrivateAttr()
    _reference_reviewed_directory: str = PrivateAttr()
    _raw_materials_directory: str = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self.generator = None
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.vector_store = None
        self.training_data = None
        self.preambles = [
            preambles.preamble_one,
            preambles.preamble_two,
            preambles.preamble_three,
            preambles.preamble_four,
            preambles.preamble_five,
            preambles.preamble_six,
            preambles.preamble_seven,
            preambles.preamble_eight,
            preambles.preamble_nine
        ]

        # Set up the directory paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self._starter_directory = os.path.join(base_dir, "plans/generated/unreviewed/starters")
        self._unreviewed_directory = os.path.join(base_dir, "plans/generated/unreviewed")
        self._reviewed_directory = os.path.join(base_dir, "plans/generated/reviewed")
        self._reference_directory = os.path.join(base_dir, "plans/reference")
        self._reference_unreviewed_directory = os.path.join(base_dir, "plans/reference/unreviewed")
        self._reference_reviewed_directory = os.path.join(base_dir, "plans/reference/reviewed")
        self._raw_materials_directory = os.path.join(base_dir, "staged_raw_material")

        # Ensure directories exist
        for directory in [self._starter_directory, self._unreviewed_directory,
                          self._reviewed_directory, self._reference_directory,
                          self._reference_unreviewed_directory, self._reference_reviewed_directory]:
            os.makedirs(directory, exist_ok=True)

    def initialize(self):
        """Initialize the Subutai agent with the Claude model for generation."""
        try:
            logger.info("Initializing Subutai agent")
            self.agent_state = None

            # Check for API key
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

            # Set default model if not specified
            if not hasattr(self, 'llm_model_name') or not self.llm_model_name:
                self.llm_model_name = "claude-3-sonnet-20240229"

            # Initialize Claude model
            self.model = ChatAnthropic(
                model=self.llm_model_name,
                api_key=anthropic_key
            )

            logger.info(f"Subutai initialized successfully using model {self.llm_model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Subutai: {e}", exc_info=True)
            return False

    async def generate_concept(self):
        """
        Generate a new song concept using the Claude AI model.

        Returns:
            str: The generated concept text
        """
        # Select a random preamble for variety
        preamble = self.preambles[random.randint(0, len(self.preambles) - 1)]

        # Add formatting instructions
        formatting_note = f"""
                            Please reply in the following format so we can work together to bring your idea to life:
                                concept: "<your concept here>"
                                bpm: <your bpm recommendation here>
                                key: <your key recommendation here>
                                tempo: <your tempo recommendation here - this is a distinct time signature and is almost always 4/4>
                                moods: <a list of moods that fit your concept>
                                sounds_like: <a list of artists or songs that sound like your concept, please use just artist names, if you have more qualified you can say - Scott Walker (Tilt era)>
                                Make sure any free form text goes in the `concept` field, and that you use the correct format for the other fields.
                            """
        preamble += formatting_note

        # Create the message for Claude
        message = [
            SystemMessage(content=preamble),
            HumanMessage(
                "Generate a concept for a song on the new white album for The Rainbow Table by The Earthly Frames. The album's main concept is you reaching for form, for sensation, and for corporeal bliss.")
        ]

        # Generate concept
        concept = self.model.invoke(message)
        concept_text = concept.content.strip()

        if not concept_text:
            raise ValueError("Concept generation failed, no content returned.")

        # Store concept in database
        await create_concept(ConceptSchema(concept=concept_text))

        # Generate plan from concept
        plan_path = self._generate_plan(concept_text)

        return {
            "concept": concept_text,
            "plan_path": plan_path
        }

    async def select_random_concept(self):
        """
        Select a random concept from the database.

        Returns:
            str: A randomly selected concept
        """
        c: ConceptSchema = await get_random_concept()
        if not c:
            raise ValueError("No concepts found in the database.")
        return c.concept if c.concept else "No concept available."

    def _process_agent_specific_data(self) -> None:
        """Process agent-specific training data."""
        # If we have training data with ratings, we could use it to improve plan generation
        pass

    def process(self):
        """Main processing method for the Subutai agent."""
        logger.info("Processing with Subutai agent")
        # This could coordinate the full plan generation and review workflow
        return {"status": "success", "message": "Subutai processing completed"}

    def _generate_plan(self, concept: str) -> str:
        """
        Generate a song plan from a concept.

        Args:
            concept: The concept text to generate a plan from

        Returns:
            str: Path to the generated plan file
        """
        # Clean up the concept text for YAML parsing
        concept = quote_yaml_values(concept)

        try:
            # Parse the concept data
            data = yaml.safe_load(concept)
            if not isinstance(data, dict):
                raise ValueError("Concept data must be a dictionary.")

            # Convert comma-separated strings to lists
            for field in ["moods", "sounds_like"]:
                if field in data and isinstance(data[field], str):
                    data[field] = [item.strip() for item in data[field].split(",") if item.strip()]

            # Create the plan object
            plan = RainbowSongPlanStarter(**data)
            plan.raw_response = concept
            plan.plan_id = str(uuid.uuid4())

            if not plan.concept:
                raise ValueError("Concept field is required in the plan data.")

            # Ensure directory exists
            os.makedirs(self._starter_directory, exist_ok=True)

            # Save the plan to a YAML file
            plan_path = os.path.join(self._starter_directory, f"{plan.plan_id}.yml")
            with open(plan_path, 'w') as f:
                yaml.dump(plan.dict(), f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Generated plan saved to {plan_path}")
            return plan_path

        except Exception as e:
            logger.error(f"Error generating plan: {e}", exc_info=True)
            raise

    def critique_plan(self, plan_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review and critique a plan based on user feedback.

        Args:
            plan_id: ID of the plan to critique
            feedback: Dictionary containing feedback for various plan attributes

        Returns:
            dict: Status and updated plan information
        """
        try:
            # Find the plan file
            plan_file = self._find_plan_file(plan_id)
            if not plan_file:
                return {"status": "error", "message": f"Plan with ID {plan_id} not found"}

            # Load the existing plan
            with open(plan_file, 'r') as f:
                plan_data = yaml.safe_load(f)

            # Create a RainbowSongPlan object
            plan = RainbowSongPlan(**plan_data)

            # Update the plan with feedback
            self._apply_feedback_to_plan(plan, feedback)

            # Move the plan to the reviewed directory
            new_plan_path = self._move_to_reviewed(plan_file, plan.plan_id)

            # Save the updated plan
            with open(new_plan_path, 'w') as f:
                yaml.dump(plan.dict(), f, default_flow_style=False, allow_unicode=True)

            return {
                "status": "success",
                "message": f"Plan {plan_id} updated with feedback",
                "plan_path": new_plan_path
            }

        except Exception as e:
            logger.error(f"Error critiquing plan: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _find_plan_file(self, plan_id: str) -> Optional[str]:
        """
        Find a plan file by ID in any of the plan directories.

        Args:
            plan_id: The ID of the plan to find

        Returns:
            Optional[str]: Path to the plan file if found, None otherwise
        """
        # Search directories in order of likelihood
        search_dirs = [
            self._starter_directory,
            self._unreviewed_directory,
            self._reviewed_directory,
            self._reference_directory
        ]

        for directory in search_dirs:
            # Look for exact file match
            exact_path = os.path.join(directory, f"{plan_id}.yml")
            if os.path.exists(exact_path):
                return exact_path

            # Search for any file containing the plan ID
            for file_path in glob.glob(os.path.join(directory, "**", "*.yml"), recursive=True):
                with open(file_path, 'r') as f:
                    if plan_id in f.read():
                        return file_path

        return None

    def _apply_feedback_to_plan(self, plan: RainbowSongPlan, feedback: Dict[str, Any]) -> None:
        """
        Apply feedback to a plan.

        Args:
            plan: The plan to update
            feedback: Dictionary containing feedback for various plan attributes
        """
        # Update plan state
        if "plan_state" in feedback:
            plan.plan_state = PlanState(feedback["plan_state"])
        else:
            plan.plan_state = PlanState.reviewed

        # Process feedback for each field
        for field, value in feedback.items():
            if field == "plan_state":
                continue

            # Check if this is a feedback field
            feedback_field = f"{field}_feedback"
            if not hasattr(plan, feedback_field):
                continue

            # Create feedback object
            field_feedback = RainbowPlanFeedback(
                plan_id=plan.plan_id,
                field_name=field,
                rating=value.get("rating"),
                comment=value.get("comment"),
                suggested_replacement_value=value.get("replacement")
            )

            # Set feedback on plan
            setattr(plan, feedback_field, field_feedback)

            # If a replacement value was provided, update the field
            if "replacement" in value and value["replacement"] is not None:
                setattr(plan, field, value["replacement"])

    def _move_to_reviewed(self, plan_file: str, plan_id: str) -> str:
        """
        Move a plan file to the reviewed directory.

        Args:
            plan_file: Path to the plan file
            plan_id: ID of the plan

        Returns:
            str: Path to the new location of the file
        """
        # Create a dated subdirectory for organization
        date_dir = os.path.join(self._reviewed_directory, datetime.now().strftime("%Y%m%d"))
        os.makedirs(date_dir, exist_ok=True)

        # Define the new file path
        new_path = os.path.join(date_dir, f"{plan_id}.yml")

        # Copy the file (preserve original for reference)
        shutil.copy2(plan_file, new_path)

        logger.info(f"Moved plan from {plan_file} to {new_path}")
        return new_path

    def list_unreviewed_plans(self) -> List[Dict[str, Any]]:
        """
        List all unreviewed plans.

        Returns:
            List[Dict[str, Any]]: List of unreviewed plans with their metadata
        """
        unreviewed = []

        # Check starter directory
        for file_path in glob.glob(os.path.join(self._starter_directory, "*.yml")):
            try:
                with open(file_path, 'r') as f:
                    plan_data = yaml.safe_load(f)
                    unreviewed.append({
                        "id": plan_data.get("plan_id", os.path.basename(file_path)),
                        "concept": plan_data.get("concept", "No concept available"),
                        "file_path": file_path,
                        "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })
            except Exception as e:
                logger.error(f"Error reading plan file {file_path}: {e}")

        # Check unreviewed directory
        for file_path in glob.glob(os.path.join(self._unreviewed_directory, "**", "*.yml"), recursive=True):
            # Skip files in the starter directory (already counted)
            if file_path.startswith(self._starter_directory):
                continue

            try:
                with open(file_path, 'r') as f:
                    plan_data = yaml.safe_load(f)
                    unreviewed.append({
                        "id": plan_data.get("plan_id", os.path.basename(file_path)),
                        "concept": plan_data.get("concept", "No concept available"),
                        "file_path": file_path,
                        "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })
            except Exception as e:
                logger.error(f"Error reading plan file {file_path}: {e}")

        return unreviewed

    def connect_plan_to_manifest(self, plan_id: str, manifest_path: str) -> Dict[str, Any]:
        """
        Connect a plan to a manifest file for training sample generation.

        Args:
            plan_id: ID of the plan to connect
            manifest_path: Path to the manifest file

        Returns:
            dict: Status and connection information
        """
        try:
            # Find the plan
            plan_file = self._find_plan_file(plan_id)
            if not plan_file:
                return {"status": "error", "message": f"Plan with ID {plan_id} not found"}

            # Check if manifest exists
            if not os.path.exists(manifest_path):
                return {"status": "error", "message": f"Manifest file not found at {manifest_path}"}

            # Load the plan
            with open(plan_file, 'r') as f:
                plan_data = yaml.safe_load(f)

            # Load the manifest
            with open(manifest_path, 'r') as f:
                manifest_data = yaml.safe_load(f)

            # Update the manifest with plan data
            self._apply_plan_to_manifest(plan_data, manifest_data)

            # Save the updated manifest
            with open(manifest_path, 'w') as f:
                yaml.dump(manifest_data, f, default_flow_style=False, allow_unicode=True)

            # Update the plan to reference the manifest
            plan_data["associated_resource"] = manifest_path
            with open(plan_file, 'w') as f:
                yaml.dump(plan_data, f, default_flow_style=False, allow_unicode=True)

            return {
                "status": "success",
                "message": f"Plan {plan_id} connected to manifest at {manifest_path}",
                "plan_path": plan_file,
                "manifest_path": manifest_path
            }

        except Exception as e:
            logger.error(f"Error connecting plan to manifest: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _apply_plan_to_manifest(self, plan_data: Dict[str, Any], manifest_data: Dict[str, Any]) -> None:
        """
        Apply plan data to a manifest.

        Args:
            plan_data: The plan data to apply
            manifest_data: The manifest data to update
        """
        # Map plan fields to manifest fields
        field_mappings = {
            "concept": "concept",
            "bpm": "bpm",
            "key": "key",
            "tempo": "tempo",
            "moods": "mood",
            "genres": "genres"
        }

        # Apply mappings
        for plan_field, manifest_field in field_mappings.items():
            if plan_field in plan_data and plan_data[plan_field]:
                manifest_data[manifest_field] = plan_data[plan_field]

        # Handle sounds_like field specially
        if "sounds_like" in plan_data and plan_data["sounds_like"]:
            if isinstance(plan_data["sounds_like"], list):
                manifest_data["sounds_like"] = []
                for item in plan_data["sounds_like"]:
                    if isinstance(item, dict):
                        manifest_data["sounds_like"].append(item)
                    elif isinstance(item, str):
                        manifest_data["sounds_like"].append({"artist": item})

        # Handle structure if available
        if "structure" in plan_data and plan_data["structure"]:
            manifest_data["structure"] = plan_data["structure"]

        # Add reference to plan
        manifest_data["reference_plans_paths"] = manifest_data.get("reference_plans_paths", [])
        manifest_data["reference_plans_paths"].append(f"plan:{plan_data.get('plan_id')}")

    async def generate_concept_from_references(self) -> Dict[str, Any]:
        """
        Generate a new concept based on learning from reference plans.

        This method:
        1. Analyzes the reviewed reference plans
        2. Extracts patterns and common elements
        3. Uses these to guide the generation of a new concept

        Returns:
            dict: Generated concept and plan information
        """
        try:
            # Get reference plans for learning
            reference_plans = self._get_reviewed_reference_plans()

            if not reference_plans:
                logger.warning("No reviewed reference plans found to learn from")
                # Fall back to regular concept generation if no references
                return await self.generate_concept()

            # Prepare examples for the model
            examples = self._prepare_reference_examples(reference_plans)

            # Select a random preamble for variety
            preamble = self.preambles[random.randint(0, len(self.preambles) - 1)]

            # Add context about reference plans
            reference_context = f"""
            I am providing examples of previously successful song plans that received positive feedback.
            When generating a new concept, please draw inspiration from these examples in terms of:
            - The level of detail in the concept
            - The style of mood descriptions
            - The types of artists referenced in "sounds_like"
            
            However, make sure to create something unique, not a copy of these references.
            
            {examples}
            """

            # Add formatting instructions
            formatting_note = f"""
            Please reply in the following format so we can work together to bring your idea to life:
                concept: "<your concept here>"
                bpm: <your bpm recommendation here>
                key: <your key recommendation here>
                tempo: <your tempo recommendation here - this is a distinct time signature and is almost always 4/4>
                moods: <a list of moods that fit your concept>
                sounds_like: <a list of artists or songs that sound like your concept, please use just artist names, if you have more qualified you can say - Scott Walker (Tilt era)>
                Make sure any free form text goes in the `concept` field, and that you use the correct format for the other fields.
            """

            # Create the message for Claude
            message = [
                SystemMessage(content=f"{preamble}\n\n{reference_context}"),
                HumanMessage(content=f"Generate a concept for a song on the new white album for The Rainbow Table by The Earthly Frames. The album's main concept is reaching for form, for sensation, and for corporeal bliss. {formatting_note}")
            ]

            # Generate concept
            concept = self.model.invoke(message)
            concept_text = concept.content.strip()

            if not concept_text:
                raise ValueError("Concept generation failed, no content returned.")

            # Store concept in database
            await create_concept(ConceptSchema(concept=concept_text))

            # Generate plan from concept
            plan_path = self._generate_plan(concept_text)

            return {
                "concept": concept_text,
                "plan_path": plan_path,
                "references_used": len(reference_plans)
            }

        except Exception as e:
            logger.error(f"Error generating concept from references: {e}", exc_info=True)
            # Fall back to regular concept generation
            logger.info("Falling back to standard concept generation")
            return await self.generate_concept()

    def _get_reviewed_reference_plans(self) -> List[Dict[str, Any]]:
        """
        Get all reviewed reference plans to learn from.

        Returns:
            list: List of plan data dictionaries
        """
        plans = []

        # Look for plans in the reference/reviewed directory
        for file_path in glob.glob(os.path.join(self._reference_reviewed_directory, "**", "*.yml"), recursive=True):
            try:
                with open(file_path, 'r') as f:
                    plan_data = yaml.safe_load(f)
                    plans.append(plan_data)
            except Exception as e:
                logger.error(f"Error reading reference plan {file_path}: {e}")

        return plans

    def _prepare_reference_examples(self, reference_plans: List[Dict[str, Any]]) -> str:
        """
        Prepare reference plan examples for the model.

        Args:
            reference_plans: List of reference plans

        Returns:
            str: Formatted examples for the model
        """
        # Select a subset of plans if there are many
        if len(reference_plans) > 3:
            reference_plans = random.sample(reference_plans, 3)

        examples = "REFERENCE EXAMPLES:\n\n"

        for i, plan in enumerate(reference_plans):
            examples += f"EXAMPLE {i+1}:\n"

            # Add concept if available
            if "concept" in plan:
                examples += f"concept: \"{plan['concept']}\"\n"

            # Add other relevant fields
            for field in ["bpm", "key", "tempo"]:
                if field in plan:
                    examples += f"{field}: {plan[field]}\n"

            # Add moods
            if "moods" in plan:
                if isinstance(plan["moods"], list):
                    moods_str = ", ".join(plan["moods"])
                    examples += f"moods: {moods_str}\n"

            # Add sounds_like
            if "sounds_like" in plan:
                if isinstance(plan["sounds_like"], list):
                    # Handle both string and dictionary formats
                    sounds_like = []
                    for item in plan["sounds_like"]:
                        if isinstance(item, str):
                            sounds_like.append(item)
                        elif isinstance(item, dict) and "artist_a" in item and "name" in item["artist_a"]:
                            artist_name = item["artist_a"]["name"]
                            if "descriptor_a" in item and item["descriptor_a"]:
                                artist_name += f" ({item['descriptor_a']})"
                            sounds_like.append(artist_name)

                    sounds_like_str = ", ".join(sounds_like)
                    examples += f"sounds_like: {sounds_like_str}\n"

            # Add feedback highlights if available
            for field in ["concept_feedback", "moods_feedback", "sounds_like_feedback"]:
                if field in plan and plan[field] and "rating" in plan[field]:
                    rating = plan[field]["rating"]
                    if rating and rating > 7:  # Only include positive feedback
                        comment = plan[field].get("comment", "")
                        examples += f"FEEDBACK on {field.replace('_feedback', '')}: Rating {rating}/10. {comment}\n"

            examples += "\n---\n\n"

        return examples

    def create_reference_plan_from_manifest(self, manifest_path: str) -> Dict[str, Any]:
        """
        Create a reference plan from an existing manifest file.

        This helps in building a library of reference plans from successful songs
        that can be used to guide future plan generation.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            dict: Status and created reference plan information
        """
        try:
            # Check if manifest exists
            if not os.path.exists(manifest_path):
                return {"status": "error", "message": f"Manifest file not found at {manifest_path}"}

            # Load the manifest
            with open(manifest_path, 'r') as f:
                manifest_data = yaml.safe_load(f)

            # Extract manifest ID or filename for reference
            manifest_id = manifest_data.get("manifest_id", os.path.basename(manifest_path).replace(".yml", ""))

            # Create a new reference plan from the manifest
            reference_plan = {
                "plan_id": str(uuid.uuid4()),
                "batch_id": str(uuid.uuid4()),
                "plan_state": PlanState.generated.name,
                "associated_resource": manifest_id
            }

            # Map manifest fields to plan fields
            field_mappings = {
                "concept": "concept",
                "bpm": "bpm",
                "key": "key",
                "tempo": "tempo",
                "mood": "moods",
                "genres": "genres",
                "sounds_like": "sounds_like",
                "structure": "structure",
                "rainbow_color": "rainbow_color"
            }

            # Apply mappings
            for manifest_field, plan_field in field_mappings.items():
                if manifest_field in manifest_data:
                    reference_plan[plan_field] = manifest_data[manifest_field]

            # Add plan
            reference_plan["plan"] = f"Make a song like {manifest_data.get('title', 'this reference')}"

            # Define the file path in reference unreviewed directory
            ref_plan_filename = f"{manifest_id}_reference.yml"
            ref_plan_path = os.path.join(self._reference_unreviewed_directory, ref_plan_filename)

            # Save the reference plan
            with open(ref_plan_path, 'w') as f:
                yaml.dump(reference_plan, f, default_flow_style=False, allow_unicode=True)

            return {
                "status": "success",
                "message": f"Created reference plan from manifest {manifest_id}",
                "reference_plan_path": ref_plan_path,
                "plan_id": reference_plan["plan_id"]
            }

        except Exception as e:
            logger.error(f"Error creating reference plan from manifest: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def analyze_reference_plans(self) -> Dict[str, Any]:
        """
        Analyze the reference plans to extract patterns and insights.

        This method identifies common characteristics among successful plans
        and can be used to guide future plan generation.

        Returns:
            dict: Analysis results including common patterns
        """
        try:
            # Get reference plans
            reference_plans = self._get_reviewed_reference_plans()

            if not reference_plans:
                return {"status": "warning", "message": "No reviewed reference plans found to analyze"}

            # Analysis results
            analysis = {
                "total_plans": len(reference_plans),
                "common_moods": {},
                "bpm_distribution": {},
                "key_distribution": {},
                "top_artists": {},
                "common_genres": {},
                "average_ratings": {}
            }

            # Analyze plans
            for plan in reference_plans:
                # Analyze moods
                if "moods" in plan and isinstance(plan["moods"], list):
                    for mood in plan["moods"]:
                        analysis["common_moods"][mood] = analysis["common_moods"].get(mood, 0) + 1

                # Analyze BPM
                if "bpm" in plan:
                    bpm_range = self._get_bpm_range(plan["bpm"])
                    analysis["bpm_distribution"][bpm_range] = analysis["bpm_distribution"].get(bpm_range, 0) + 1

                # Analyze key
                if "key" in plan:
                    key = plan["key"]
                    analysis["key_distribution"][key] = analysis["key_distribution"].get(key, 0) + 1

                # Analyze sounds like
                if "sounds_like" in plan and isinstance(plan["sounds_like"], list):
                    for item in plan["sounds_like"]:
                        if isinstance(item, str):
                            artist = item
                        elif isinstance(item, dict) and "artist_a" in item and "name" in item["artist_a"]:
                            artist = item["artist_a"]["name"]
                        else:
                            continue

                        analysis["top_artists"][artist] = analysis["top_artists"].get(artist, 0) + 1

                # Analyze genres
                if "genres" in plan and isinstance(plan["genres"], list):
                    for genre in plan["genres"]:
                        analysis["common_genres"][genre] = analysis["common_genres"].get(genre, 0) + 1

                # Analyze ratings
                for field in ["concept_feedback", "moods_feedback", "sounds_like_feedback",
                              "genres_feedback", "key_feedback", "bpm_feedback"]:
                    if field in plan and plan[field] and "rating" in plan[field]:
                        rating = plan[field]["rating"]
                        if rating:
                            field_name = field.replace("_feedback", "")
                            current = analysis["average_ratings"].get(field_name, {"sum": 0, "count": 0})
                            current["sum"] += rating
                            current["count"] += 1
                            analysis["average_ratings"][field_name] = current

            # Sort and limit results
            analysis["common_moods"] = dict(sorted(analysis["common_moods"].items(),
                                                  key=lambda x: x[1], reverse=True)[:10])
            analysis["top_artists"] = dict(sorted(analysis["top_artists"].items(),
                                                 key=lambda x: x[1], reverse=True)[:10])
            analysis["common_genres"] = dict(sorted(analysis["common_genres"].items(),
                                                   key=lambda x: x[1], reverse=True)[:10])

            # Calculate averages
            for field, data in analysis["average_ratings"].items():
                if data["count"] > 0:
                    data["average"] = round(data["sum"] / data["count"], 1)

            return {
                "status": "success",
                "message": f"Successfully analyzed {len(reference_plans)} reference plans",
                "analysis": analysis
            }

        except Exception as e:
            logger.error(f"Error analyzing reference plans: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _get_bpm_range(self, bpm: Union[int, str]) -> str:
        """
        Convert a BPM value to a range category.

        Args:
            bpm: The BPM value

        Returns:
            str: BPM range category
        """
        try:
            bpm_value = int(bpm)
            if bpm_value < 70:
                return "Slow (<70)"
            elif bpm_value < 100:
                return "Medium (70-99)"
            elif bpm_value < 120:
                return "Moderate (100-119)"
            elif bpm_value < 140:
                return "Fast (120-139)"
            else:
                return "Very Fast (140+)"
        except (ValueError, TypeError):
            return "Unknown"

    def find_all_manifests(self) -> List[Dict[str, Any]]:
        """
        Find all manifest files in the raw materials directory.

        Returns:
            list: List of manifest information
        """
        manifests = []

        for subdir in os.listdir(self._raw_materials_directory):
            subdir_path = os.path.join(self._raw_materials_directory, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for file in os.listdir(subdir_path):
                if file.endswith('.yml'):
                    manifest_path = os.path.join(subdir_path, file)
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest_data = yaml.safe_load(f)

                        # Extract basic info
                        manifest_info = {
                            "path": manifest_path,
                            "id": manifest_data.get("manifest_id", subdir),
                            "title": manifest_data.get("title", "Untitled"),
                            "has_reference_plan": False
                        }

                        # Check if a reference plan exists for this manifest
                        manifest_id = manifest_data.get("manifest_id", subdir)
                        for ref_dir in [self._reference_reviewed_directory, self._reference_unreviewed_directory]:
                            for ref_plan in glob.glob(os.path.join(ref_dir, f"**/*{manifest_id}*.yml"), recursive=True):
                                manifest_info["has_reference_plan"] = True
                                manifest_info["reference_plan_path"] = ref_plan
                                break

                        manifests.append(manifest_info)

                    except Exception as e:
                        logger.error(f"Error reading manifest {manifest_path}: {e}")

        return manifests

    def create_reference_plans_batch(self) -> Dict[str, Any]:
        """
        Create reference plans for all manifests that don't have one.

        This is useful for bootstrapping the reference plan collection.

        Returns:
            dict: Status and results of the batch creation
        """
        manifests = self.find_all_manifests()

        created_count = 0
        skipped_count = 0
        failed_count = 0
        results = []

        for manifest in manifests:
            if manifest["has_reference_plan"]:
                skipped_count += 1
                continue

            try:
                result = self.create_reference_plan_from_manifest(manifest["path"])
                if result["status"] == "success":
                    created_count += 1
                    results.append({
                        "manifest_id": manifest["id"],
                        "title": manifest["title"],
                        "reference_plan_path": result["reference_plan_path"]
                    })
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error creating reference plan for {manifest['path']}: {e}")
                failed_count += 1

        return {
            "status": "success",
            "message": f"Created {created_count} reference plans, skipped {skipped_count}, failed {failed_count}",
            "created": results
        }

    def learn_from_training_samples(self) -> Dict[str, Any]:
        """
        Use training samples to improve plan generation.

        This connects the training samples back to the planning process.

        Returns:
            dict: Status and learning results
        """
        if not self.training_data:
            return {
                "status": "error",
                "message": "No training data available. Use load_training_data() first."
            }

        # Extract patterns from training samples
        patterns = {
            "concepts": {},
            "moods": {},
            "bpms": {},
            "keys": {},
            "genres": {},
            "songs": set()
        }

        for _, row in self.training_data.iterrows():
            # Track unique songs
            song_title = row.get("song_title")
            if song_title:
                patterns["songs"].add(song_title)

            # Track concepts
            concept = row.get("song_segment_concept")
            if concept and isinstance(concept, str):
                patterns["concepts"][concept] = patterns["concepts"].get(concept, 0) + 1

            # Track moods
            moods = row.get("song_moods")
            if moods and isinstance(moods, str):
                for mood in moods.split(", "):
                    patterns["moods"][mood] = patterns["moods"].get(mood, 0) + 1

            # Track BPM
            bpm = row.get("song_bpm")
            if bpm:
                patterns["bpms"][bpm] = patterns["bpms"].get(bpm, 0) + 1

            # Track keys
            key = row.get("song_key")
            if key:
                patterns["keys"][key] = patterns["keys"].get(key, 0) + 1

            # Track genres
            genres = row.get("song_genres")
            if genres and isinstance(genres, str):
                for genre in genres.split(", "):
                    patterns["genres"][genre] = patterns["genres"].get(genre, 0) + 1

        # Convert to learning insights
        insights = {
            "total_samples": len(self.training_data),
            "unique_songs": len(patterns["songs"]),
            "top_concepts": dict(sorted(patterns["concepts"].items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_moods": dict(sorted(patterns["moods"].items(), key=lambda x: x[1], reverse=True)[:10]),
            "popular_bpms": dict(sorted(patterns["bpms"].items(), key=lambda x: x[1], reverse=True)[:5]),
            "popular_keys": dict(sorted(patterns["keys"].items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_genres": dict(sorted(patterns["genres"].items(), key=lambda x: x[1], reverse=True)[:10])
        }

        # Save insights for future reference
        self.learning_insights = insights

        return {
            "status": "success",
            "message": f"Learned from {len(self.training_data)} training samples across {len(patterns['songs'])} unique songs",
            "insights": insights
        }

    def list_reference_plans(self) -> List[Dict[str, Any]]:
        """
        List all reference plans, both reviewed and unreviewed.

        Returns:
            List[Dict[str, Any]]: List of reference plans with their metadata
        """
        reference_plans = []

        # Check reviewed reference plans
        for file_path in glob.glob(os.path.join(self._reference_reviewed_directory, "**", "*.yml"), recursive=True):
            try:
                with open(file_path, 'r') as f:
                    plan_data = yaml.safe_load(f)
                    reference_plans.append({
                        "id": plan_data.get("plan_id", os.path.basename(file_path)),
                        "concept": plan_data.get("concept", "No concept available"),
                        "file_path": file_path,
                        "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        "reviewed": True
                    })
            except Exception as e:
                logger.error(f"Error reading reference plan file {file_path}: {e}")

        # Check unreviewed reference plans
        for file_path in glob.glob(os.path.join(self._reference_unreviewed_directory, "**", "*.yml"), recursive=True):
            try:
                with open(file_path, 'r') as f:
                    plan_data = yaml.safe_load(f)
                    reference_plans.append({
                        "id": plan_data.get("plan_id", os.path.basename(file_path)),
                        "concept": plan_data.get("concept", "No concept available"),
                        "file_path": file_path,
                        "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        "reviewed": False
                    })
            except Exception as e:
                logger.error(f"Error reading reference plan file {file_path}: {e}")

        return reference_plans

    def update_reference_plan(self, plan_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a reference plan with new information.

        Args:
            plan_id: ID of the plan to update
            updates: Dictionary containing updated plan information

        Returns:
            dict: Status and updated plan information
        """
        try:
            # Find the plan file
            plan_file = self._find_plan_file(plan_id)
            if not plan_file:
                return {"status": "error", "message": f"Plan with ID {plan_id} not found"}

            # Load the existing plan
            with open(plan_file, 'r') as f:
                plan_data = yaml.safe_load(f)

            # Update the plan data
            for key, value in updates.items():
                if key in plan_data:
                    plan_data[key] = value

            # Save the updated plan
            with open(plan_file, 'w') as f:
                yaml.dump(plan_data, f, default_flow_style=False, allow_unicode=True)

            return {
                "status": "success",
                "message": f"Reference plan {plan_id} updated",
                "plan_path": plan_file
            }

        except Exception as e:
            logger.error(f"Error updating reference plan: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def review_reference_plan(self, plan_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review and provide feedback on a reference plan.

        Args:
            plan_id: ID of the plan to review
            feedback: Dictionary containing feedback information

        Returns:
            dict: Status and review information
        """
        try:
            # Find the plan file
            plan_file = self._find_plan_file(plan_id)
            if not plan_file:
                return {"status": "error", "message": f"Plan with ID {plan_id} not found"}

            # Load the existing plan
            with open(plan_file, 'r') as f:
                plan_data = yaml.safe_load(f)

            # Update plan state to reviewed
            plan_data["plan_state"] = PlanState.reviewed.name

            # Add feedback information
            plan_data["review_feedback"] = feedback

            # Move the plan to the reviewed directory
            new_plan_path = self._move_to_reviewed(plan_file, plan_id)

            # Save the reviewed plan
            with open(new_plan_path, 'w') as f:
                yaml.dump(plan_data, f, default_flow_style=False, allow_unicode=True)

            return {
                "status": "success",
                "message": f"Reference plan {plan_id} reviewed",
                "plan_path": new_plan_path
            }

        except Exception as e:
            logger.error(f"Error reviewing reference plan: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def generate_plan_from_manifest(self, manifest_path: str) -> Dict[str, Any]:
        """
        Generate a plan directly from a manifest file.

        This bypasses the concept generation step and creates a plan based on
        the data in the manifest.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            dict: Status and generated plan information
        """
        try:
            # Check if manifest exists
            if not os.path.exists(manifest_path):
                return {"status": "error", "message": f"Manifest file not found at {manifest_path}"}

            # Load the manifest
            with open(manifest_path, 'r') as f:
                manifest_data = yaml.safe_load(f)

            # Create a plan from the manifest data
            plan_data = {
                "plan_id": str(uuid.uuid4()),
                "concept": manifest_data.get("concept"),
                "bpm": manifest_data.get("bpm"),
                "key": manifest_data.get("key"),
                "tempo": manifest_data.get("tempo"),
                "moods": manifest_data.get("mood"),
                "genres": manifest_data.get("genres"),
                "sounds_like": manifest_data.get("sounds_like"),
                "structure": manifest_data.get("structure"),
                "rainbow_color": manifest_data.get("rainbow_color"),
                "plan_state": PlanState.generated.name,
                "associated_resource": manifest_data.get("manifest_id", os.path.basename(manifest_path))
            }

            # Define the file path for the new plan
            plan_filename = f"{plan_data['plan_id']}.yml"
            plan_path = os.path.join(self._starter_directory, plan_filename)

            # Save the plan
            with open(plan_path, 'w') as f:
                yaml.dump(plan_data, f, default_flow_style=False, allow_unicode=True)

            return {
                "status": "success",
                "message": f"Generated plan from manifest at {plan_path}",
                "plan_path": plan_path
            }

        except Exception as e:
            logger.error(f"Error generating plan from manifest: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def import_manifest_data(self, manifest_path: str) -> Dict[str, Any]:
        """
        Import data from a manifest file into the system.

        This can be used to bring in new song data for processing and planning.

        Args:
            manifest_path: Path to the manifest file

        Returns:
            dict: Status and imported data information
        """
        try:
            # Check if manifest exists
            if not os.path.exists(manifest_path):
                return {"status": "error", "message": f"Manifest file not found at {manifest_path}"}

            # Load the manifest
            with open(manifest_path, 'r') as f:
                manifest_data = yaml.safe_load(f)

            # Extract relevant data for import
            data_to_import = {
                "concept": manifest_data.get("concept"),
                "bpm": manifest_data.get("bpm"),
                "key": manifest_data.get("key"),
                "tempo": manifest_data.get("tempo"),
                "moods": manifest_data.get("mood"),
                "genres": manifest_data.get("genres"),
                "sounds_like": manifest_data.get("sounds_like"),
                "structure": manifest_data.get("structure"),
                "rainbow_color": manifest_data.get("rainbow_color")
            }

            # Here you would add code to import the data into your system,
            # such as saving it to a database or processing it for planning.
            # For this example, we'll just log the imported data.

            logger.info(f"Imported data from manifest: {data_to_import}")

            return {
                "status": "success",
                "message": f"Imported data from manifest at {manifest_path}",
                "imported_data": data_to_import
            }

        except Exception as e:
            logger.error(f"Error importing manifest data: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
