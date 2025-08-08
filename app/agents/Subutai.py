import os
import random
import yaml
import uuid

from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from typing import Any, Optional
from pydantic import PrivateAttr

from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.objects.db_models.concept_schema import ConceptSchema
from app.resources.prompts.concepts import preambles
from app.utils.db_util import create_concept, get_random_concept
from app.objects.song_plan import RainbowSongPlanStarter
from app.utils.string_util import quote_yaml_values

load_dotenv(verbose=True)

class Subutai(BaseRainbowAgent):

    generator: Any = None
    model: Optional[Any] = None
    tokenizer: Any = None
    plan_data: Any = None
    vector_store: Any = None
    preambles: list[str] = []
    _starter_directory: str = PrivateAttr()


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
        self._starter_directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "plans/generated/unreviewed/starters"
        )

    def initialize(self):
        self.agent_state = None
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        self.model = ChatAnthropic(
            model=self.llm_model_name,
            api_key=anthropic_key)

    async def generate_concept(self):
        preamble = self.preambles[random.randint(0, len(self.preambles)-1)]
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
        message = [
            SystemMessage(content=preamble),
            HumanMessage("Generate a concept for a song on the new white album for The Rainbow Table by The Earthly Frames. The album's main concept is you reaching for form, for sensation, and for corporeal bliss.")
        ]
        concept = self.model.invoke(message)
        concept_text = concept.content.strip()
        if not concept_text:
            raise ValueError("Concept generation failed, no content returned.")
        await create_concept(ConceptSchema(concept=concept_text))
        self._generate_plan(concept_text)

    async def select_random_concept(self):
        c: ConceptSchema = await get_random_concept()
        if not c:
            raise ValueError("No concepts found in the database.")
        return c.concept if c.concept else "No concept available."

    def _process_agent_specific_data(self) -> None:
        pass

    def process(self):
        pass

    def _generate_plan(self, concept: str):
        concept = quote_yaml_values(concept)
        data = yaml.safe_load(concept)
        if not isinstance(data, dict):
            raise ValueError("Concept data must be a dictionary.")
        for field in ["moods", "sounds_like"]:
            if field in data and isinstance(data[field], str):
                data[field] = [item.strip() for item in data[field].split(",") if item.strip()]
        plan = RainbowSongPlanStarter(**data)
        plan.raw_response = concept
        plan.plan_id = str(uuid.uuid4())
        if not plan.concept:
            raise ValueError("Concept field is required in the plan data.")
        os.makedirs(self._starter_directory, exist_ok=True)
        plan_path = os.path.join(self._starter_directory, f"{plan.plan_id}.yml")
        with open(plan_path, 'w') as f:
            yaml.dump(plan.dict(), f, default_flow_style=False, allow_unicode=True)
        if os.path.exists(plan_path):
            print(f"File exists: {plan_path}")
        else:
            print(f"File NOT found: {plan_path}")

    def critique_plan(self):
        pass
