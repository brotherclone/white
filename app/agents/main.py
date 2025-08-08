import torch
import asyncio

from app.agents.Dorthy import Dorthy
from app.agents.Subutai import Subutai

TRAINING_PATH = "/Volumes/LucidNonsense/White/training"

def try_agents():
    print("Initializing Dorthy agent...")
    dorthy = Dorthy(
        tokenizer_name="gpt2",
        llm_model_name="gpt2",
        generator_name="gpt2"
    )

    print("Loading training data...")
    dorthy.load_training_data(TRAINING_PATH)

    print("Initializing model...")
    dorthy.initialize()

    if dorthy.training_data is not None and not dorthy.training_data.empty:
        sample_mood = dorthy.training_data['song_moods'].iloc[0].split(',')[0].strip()
        sample_color = dorthy.training_data['album_rainbow_color'].iloc[0]
        genre = dorthy.training_data['song_genres'].iloc[0].split(',')[0].strip()

        print(f"Testing lyric generation with mood: {sample_mood}, color: {sample_color}")
        prompt = f"Write original lyrics for a {sample_mood} {genre} song with {sample_color} color imagery:\n\n"

        try:
            result = dorthy.generator(prompt,
                                      max_new_tokens=150,
                                      num_return_sequences=1,
                                      temperature=0.8,
                                      top_p=0.9,
                                      do_sample=True,
                                      truncation=True)
            lyrics = result[0]['generated_text'].replace(prompt, "")
            print("\nGenerated lyrics:")
            print(lyrics)
        except Exception as e:
            print(f"Error generating lyrics: {e}")
    else:
        print("No training data available for Dorthy agent.")

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    s= Subutai(
        llm_model_name="claude-3-5-sonnet-latest",
    )
    s.initialize()
    con = asyncio.run(s.generate_concept())

