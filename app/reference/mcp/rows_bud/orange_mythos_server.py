import os

from typing import List, Optional, cast, Iterable, Any
from anthropic import Anthropic
from fastmcp import FastMCP
from app.reference.mcp.rows_bud.orange_corpus import get_corpus

mcp = FastMCP("Orange Mythos")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CORPUS_DIR = os.getenv("ORANGE_CORPUS_DIR")
corpus = get_corpus(CORPUS_DIR)


@mcp.tool()
def add_story_to_corpus(
    headline: str, date: str, source: str, text: str, location: str, tags: List[str]
) -> dict:
    """
    Add a mythologizable story to the corpus with automatic scoring.

    Args:
        headline: Story headline
        date: Publication date (YYYY-MM-DD)
        source: Newspaper name
        text: Full article text
        location: Specific NJ location
        tags: Category tags (e.g., ["rock_bands", "youth_crime"])

    Returns:
        dict with story_id, score, and status
    """
    try:
        story_id, score = corpus.add_story(
            headline=headline,
            date=date,
            source=source,
            text=text,
            location=location,
            tags=tags,
        )

        return {"story_id": story_id, "score": score, "status": "added"}

    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool()
def insert_symbolic_object(
    story_id: str, object_category: str, custom_object: Optional[str] = None
) -> dict:
    """
    Insert a symbolic object into a story for mythologization.

    The object should NOT exist in the original story but should be:
    - Contextually appropriate to time/place
    - Metaphorically resonant with themes
    - Mythologically significant

    Args:
        story_id: ID of a story to mythologize
        object_category: One of: CIRCULAR_TIME, INFORMATION_ARTIFACTS,
                        LIMINAL_OBJECTS, PSYCHOGEOGRAPHIC
        custom_object: Optional custom object description

    Returns:
        dict with an updated story and status
    """
    try:
        story = corpus.get_story(story_id)

        if not story:
            return {"error": f"Story {story_id} not found", "status": "error"}

        if custom_object:
            object_desc = custom_object
        else:
            object_templates = {
                "CIRCULAR_TIME": "a clock that runs at an unusual speed",
                "INFORMATION_ARTIFACTS": "a mysterious recording or transmission",
                "LIMINAL_OBJECTS": "a doorway or threshold to elsewhere",
                "PSYCHOGEOGRAPHIC": "a map with impossible coordinates",
            }
            object_desc = object_templates.get(object_category, "a strange artifact")
        prompt = f"""Insert this symbolic object into the story naturally and seamlessly.

        ORIGINAL STORY:
        {story['text']}
        
        SYMBOLIC OBJECT: {object_desc}
        CATEGORY: {object_category}
        
        CRITICAL RULES:
        - The object did NOT exist in the original story
        - Insert it as if it was always there and was discovered/noticed
        - Make it feel central to the narrative, not forced
        - Keep the journalistic tone (this is pre-gonzo rewriting)
        - The object should raise questions, create mystery
        - It should feel like a detail the original journalist might have overlooked or downplayed
        
        LOCATION CONTEXT: {story['location']} in {story['date']}
        
        Return ONLY the updated story text with the object naturally integrated. 
        Do not add any preamble or explanation."""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=cast(Iterable[Any], [{"role": "user", "content": prompt}]),
        )

        updated_text = response.content[0].text.strip()

        # Update corpus
        corpus.insert_symbolic_object(
            story_id=story_id,
            category=object_category,
            description=object_desc,
            updated_text=updated_text,
        )

        # Return updated story
        updated_story = corpus.get_story(story_id)

        return {"updated_story": updated_story, "status": "object_inserted"}

    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool()
def gonzo_rewrite(
    story_id: str, perspective: str = "journalist", intensity: int = 3
) -> dict:
    """
    Rewrite a mythologized story in gonzo journalism style.

    Style characteristics:
    - First-person embedded journalism
    - Paranoia and conspiracy undertones
    - Perception shifts and altered states
    - Authority distrust
    - Vivid sensory details
    - The symbolic object becomes central

    Args:
        story_id: ID of mythologized story
        perspective: Narrative perspective (journalist, witness, investigator, participant)
        intensity: Gonzo intensity level (1-5, where 5 is full Hunter S. Thompson)

    Returns:
        dict with gonzo story and status
    """
    try:
        story = corpus.get_story(story_id)
        if not story:
            return {"error": f"Story {story_id} not found", "status": "error"}
        intensity_styles = {
            1: "Subtle first-person observer with mild questioning of official narrative",
            2: "Embedded journalist noticing inconsistencies, growing suspicion",
            3: "Active participant, perception shifts begin, reality feels slippery",
            4: "Deep paranoia, conspiracy emerging, boundaries dissolving",
            5: "Full Hunter S. Thompson - no holds barred, reality completely unhinged",
        }
        obj_desc = story.get("symbolic_object_desc", "a mysterious object")
        prompt = f"""Rewrite this Sussex County story in gonzo journalism style.

        ORIGINAL STORY (with symbolic object already inserted):
        Headline: {story['headline']}
        Date: {story['date']}
        Location: {story['location']}
        Source: {story['source']}
        
        {story['text']}
        
        SYMBOLIC OBJECT (central to your rewrite): {obj_desc}
        
        GONZO PARAMETERS:
        - Perspective: {perspective} (first-person, embedded in the scene)
        - Intensity: {intensity}/5 - {intensity_styles.get(intensity, intensity_styles[3])}
        
        GONZO JOURNALISM CHARACTERISTICS (Hunter S. Thompson method):
        1. FIRST-PERSON EMBEDDED: You are THERE, in {story['location']}, investigating this story
        2. PARANOIA & CONSPIRACY: Official story doesn't add up, authorities are hiding something
        3. PERCEPTION SHIFTS: Reality feels unstable, witnesses contradict each other, time behaves strangely
        4. AUTHORITY DISTRUST: Police chiefs lie, officials obscure truth, something darker underneath
        5. VIVID SENSORY: Pine smell, electronic hum, the weight of {obj_desc}, sounds that shouldn't exist
        6. SYMBOLIC OBJECT CENTRAL: {obj_desc} is THE KEY - it's proof, it's evidence, it's WRONG
        7. INVESTIGATOR BECOMES PART OF STORY: Boundary between observer and participant dissolves
        8. SUSSEX COUNTY MYTHOLOGY: This is New Jersey gothic, Pine Barrens energy, teenage doom
        
        The date is {story['date']}. You're investigating what happened. The more you dig, the weirder it gets.
        {obj_desc} keeps appearing, impossible but undeniable. What's happening in {story['location']}?
        
        CRITICAL: Keep the original factual skeleton (who, what, when, where) but:
        - Add your first-person investigation
        - Show how the official story breaks down
        - Make {obj_desc} central and inexplicable
        - End with you realizing you're part of the transmission
        
        At intensity {intensity}, go {"absolutely wild - full Thompson madness" if intensity >= 4 else "strange but controlled" if intensity <= 2 else "deep into conspiracy territory"}.
        
        Return ONLY the gonzo rewritten story. No preamble, no explanation. Make it legendary."""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.8 + (intensity * 0.05),  # Higher temp for higher intensity
            messages=cast(Iterable[Any], [{"role": "user", "content": prompt}]),
        )

        gonzo_text = response.content[0].text.strip()
        corpus.add_gonzo_rewrite(
            story_id=story_id,
            gonzo_text=gonzo_text,
            perspective=perspective,
            intensity=intensity,
        )
        updated_story = corpus.get_story(story_id)
        gonzo_story = {**updated_story, "text": gonzo_text}

        return {"gonzo_story": gonzo_story, "status": "gonzo_complete"}

    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool()
def search_corpus(
    tags: Optional[List[str]] = None,
    min_score: float = 0.5,
    location: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    needs_mythologizing: bool = True,
) -> dict:
    """
    Search the mythologizable story corpus.

    Args:
        tags: Required tags (e.g., ["rock_bands", "youth_crime"])
        min_score: Minimum mythologization score (0.0-1.0)
        location: Specific NJ location filter
        start_date: Date range start (YYYY-MM-DD)
        end_date: Date range end (YYYY-MM-DD)
        needs_mythologizing: Skip already mythologized stories (default true)

    Returns:
        dict with matching stories
    """
    try:
        results = corpus.search(
            tags=tags,
            min_score=min_score,
            location=location,
            start_date=start_date,
            end_date=end_date,
            needs_mythologizing=needs_mythologizing,
        )

        return {"results": results, "count": len(results), "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool()
def get_corpus_stats() -> dict:
    """
    Get statistics about the mythologizable story corpus.

    Returns:
        - Total stories
        - Date range coverage
        - Tag distribution
        - Mythologization status
        - Average scores
    """
    try:
        return corpus.get_stats()

    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool()
def export_corpus_json(filename: str = "corpus_export.json") -> dict:
    """
    Export the entire corpus as JSON for human inspection.

    Args:
        filename: Output filename (saved in exports/ directory)

    Returns:
        dict with export path and status
    """
    try:
        export_path = corpus.export_json(filename)

        return {"export_path": export_path, "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool()
def export_for_training(filename: str = "orange_corpus_training.parquet") -> dict:
    """
    Export mythologized stories for the training pipeline.

    Only includes stories with both original text and gonzo version.
    Output format is parquet for easy integration with training scripts.

    Args:
        filename: Output filename (saved in exports/ directory)

    Returns:
        dict with export path, story count, and status
    """
    try:
        export_path = corpus.export_for_training(filename)

        # Get count of exported stories
        import polars as pl

        training_df = pl.read_parquet(export_path)

        return {
            "export_path": export_path,
            "story_count": len(training_df),
            "status": "success",
        }

    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool()
def get_high_value_stories(limit: int = 10) -> dict:
    """
    Get top mythologizable stories that haven't been gonzo'd yet.

    Useful for finding the best candidates for mythologization.

    Args:
        limit: Maximum number of stories to return (default 10)

    Returns:
        dict with high-value stories sorted by score
    """
    try:
        stories = corpus.get_high_value_stories(limit=limit)

        return {"stories": stories, "count": len(stories), "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "error"}


@mcp.tool()
def analyze_temporal_patterns() -> dict:
    """
    Analyze when mysterious events cluster (for mythology mining).

    Returns yearly distribution, peak years, and highest-scoring periods.
    """
    try:
        patterns = corpus.analyze_temporal_patterns()

        return {**patterns, "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    print("🌹 Orange Mythos MCP Server (Polars Edition)")
    print("   Sussex County mythologizer - 182 BPM transmission")
    print(f"   Corpus: {CORPUS_DIR}")
    print(f"   Stories loaded: {len(corpus.df)}")
    mcp.run()
