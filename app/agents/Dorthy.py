import random
import pandas as pd
from typing import Any, Dict

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
        self.llm_model_name = "gpt2"
        self.tokenizer_name = "gpt2"
        self.generator_name = "gpt2"

    def initialize(self):
        self.agent_state = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModel.from_pretrained(
            self.llm_model_name,
            use_safetensors=True
        )
        self.generator = pipeline("text-generation", model=self.generator_name)

    def _process_agent_specific_data(self) -> None:
        if self.training_data is None:
            print("No training data available to process.")
            return
        print(self.training_data.columns)

    def process(self):
        # Processing logic specific to Dorthy
        pass


    def generate_lyrics_from_training_data(self,
                                           target_mood: list[str],
                                           rainbow_color: str,
                                           section_name: str,
                                           key: str = None) -> Dict[str, Any]:
        """Generate lyrics based on your actual training data"""

        print(f"🎤 Generating lyrics for {section_name} with mood: {target_mood}")

        # Find similar segments from your training data
        similar_segments = self.find_similar_segments(
            target_mood=target_mood,
            rainbow_color=rainbow_color,
            section_type=section_name,
            target_key=key,
            limit=20
        )

        if similar_segments.empty:
            print("⚠️ No similar segments found, using fallback")
            return self._fallback_lyrics(section_name)

        print(f"📚 Found {len(similar_segments)} similar segments")

        # Extract lyrical patterns
        lyrical_content = []
        for _, segment in similar_segments.iterrows():
            if pd.notna(segment.get('song_segment_lyrics_text')):
                lyrics = segment['song_segment_lyrics_text']
                if lyrics and len(lyrics.strip()) > 10:
                    lyrical_content.append({
                        'lyrics': lyrics,
                        'song_title': segment.get('song_title', 'Unknown'),
                        'mood_score': segment.get('mood_score', 0)
                    })

        if not lyrical_content:
            print("⚠️ No lyrics found in similar segments")
            return self._fallback_lyrics(section_name)

        # Use vector store if available for more sophisticated matching
        if self.vector_store:
            return self._generate_with_vector_store(lyrical_content, target_mood, section_name)
        else:
            return self._generate_with_pattern_matching(lyrical_content, target_mood, section_name)

    def _generate_with_vector_store(self, lyrical_content: list[Dict],
                                    target_mood: list[str],
                                    section_name: str) -> Dict[str, Any]:
        """Generate lyrics using vector similarity"""

        # Query vector store with mood keywords
        mood_query = " ".join(target_mood)

        try:
            # Get similar lyrics from vector store
            similar_docs = self.vector_store.similarity_search(mood_query, k=5)

            # Combine with training data patterns
            inspiration_lyrics = []
            for doc in similar_docs:
                inspiration_lyrics.append(doc.page_content)

            # For now, select the best matching lyrics as inspiration
            # In a full implementation, you'd use a language model here
            best_match = lyrical_content[0] if lyrical_content else None

            return {
                'generated_lyrics': best_match['lyrics'] if best_match else "Generated lyrics based on training data",
                'inspiration_source': best_match['song_title'] if best_match else "Training data",
                'confidence': best_match['mood_score'] if best_match else 0.5,
                'method': 'vector_store',
                'similar_examples': len(similar_docs)
            }

        except Exception as e:
            print(f"Vector store error: {e}")
            return self._generate_with_pattern_matching(lyrical_content, target_mood, section_name)

    def _generate_with_pattern_matching(self, lyrical_content: list[Dict],
                                        target_mood: list[str],
                                        section_name: str) -> Dict[str, Any]:
        """Generate lyrics using pattern matching from your training data"""

        # Sort by mood score and select best matches
        lyrical_content.sort(key=lambda x: x['mood_score'], reverse=True)

        # Take the top 3 as inspiration
        top_matches = lyrical_content[:3]

        # For now, use the best match as a template
        # In full implementation, you'd create new lyrics inspired by patterns
        if top_matches:
            best_match = top_matches[0]

            # Simple pattern-based generation (you'd enhance this)
            generated = self._create_variation(best_match['lyrics'], target_mood, section_name)

            return {
                'generated_lyrics': generated,
                'inspiration_source': best_match['song_title'],
                'confidence': best_match['mood_score'],
                'method': 'pattern_matching',
                'training_examples_used': len(top_matches)
            }

        return self._fallback_lyrics(section_name)

    @staticmethod
    def _create_variation(original_lyrics: str, target_mood: list[str], section_name: str) -> str:
        """Create a variation based on original lyrics from your training data"""

        # Simple variation for now - in full implementation you'd use more sophisticated methods
        lines = original_lyrics.strip().split('\n')

        # Add mood-specific modifications
        mood_str = ' '.join(target_mood).lower()

        if 'dark' in mood_str or 'mysterious' in mood_str:
            # Add darker imagery
            variations = [
                "In shadows deep and mysteries untold",
                "Where darkness whispers secrets of the night",
                "Through veils of time and forgotten dreams"
            ]
        elif 'ethereal' in mood_str or 'otherworldly' in mood_str:
            variations = [
                "Beyond the realm of mortal understanding",
                "In spaces between the stars and dreams",
                "Where reality bends and spirits dance"
            ]
        else:
            # Use original as base
            return original_lyrics

        # Simple variation - replace first line with mood-appropriate content
        if lines:
            lines[0] = random.choice(variations)

        return '\n'.join(lines)

    @staticmethod
    def _fallback_lyrics(section_name: str) -> Dict[str, Any]:
        """Fallback when no training data available"""
        return {
            'generated_lyrics': f"[{section_name} lyrics to be generated]",
            'inspiration_source': 'fallback',
            'confidence': 0.1,
            'method': 'fallback',
            'training_examples_used': 0
        }