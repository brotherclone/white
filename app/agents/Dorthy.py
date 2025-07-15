from typing import Any

from transformers import AutoTokenizer, AutoModel, pipeline
from app.agents.BaseRainbowAgent import BaseRainbowAgent

class Dorthy(BaseRainbowAgent):

    generator: Any = None
    model: Any = None
    tokenizer: Any = None
    lyrics_data: Any = None
    vector_store: Any = None

    def __init__(self, **data):
        super().__init__(**data)
        self.generator = None
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.vector_store = None
        self.training_data = None

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(self.llm_model_name)
        self.generator = pipeline("text-generation", model=self.generator_name)

    def _process_agent_specific_data(self) -> None:
        if self.training_data is None:
            print("No training data available to process.")
            return
        self.lyrics_data = self.training_data[
            ['song_segment_lyrics_text', 'song_moods', 'album_rainbow_color',
             'song_segment_description', 'song_genres']
        ].dropna(subset=['song_segment_lyrics_text'])
        self.create_vector_store(
            text_field='song_segment_lyrics_text',
            metadata_fields=['song_moods', 'album_rainbow_color', 'song_genres']
        )
        print(f"Processed {len(self.lyrics_data)} lyric samples for Dorthy")

    def process(self):
        # Processing logic specific to Dorthy
        pass
