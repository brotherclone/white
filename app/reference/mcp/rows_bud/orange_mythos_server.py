#!/usr/bin/env python3
"""
Orange Agent MCP Server - Mytho-Temporal Rebracketing Engine

Provides tools for:
- Searching Internet Archive newspapers
- Building mythologizable story corpus
- Inserting symbolic objects
- Gonzo-style narrative transformation
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# MCP SDK
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
)
import mcp.server.stdio

# Internet Archive
try:
    from internetarchive import search_items

    IA_AVAILABLE = True
except ImportError:
    IA_AVAILABLE = False
    print("Warning: internetarchive not installed. Run: pip install internetarchive")

# Storage
CORPUS_DIR = Path(os.getenv("ORANGE_CORPUS_DIR", "./orange_mythos_corpus"))
CORPUS_FILE = CORPUS_DIR / "mythologizable_corpus.json"

# Symbolic object categories
SYMBOLIC_OBJECTS = {
    "CIRCULAR_TIME": [
        "Broken clock stuck at {time}",
        "Möbius strip made of {material}",
        "Mirror showing {temporal_variant}",
        "Photograph that changes over time",
        "Cassette tape that plays backwards",
    ],
    "INFORMATION_ARTIFACTS": [
        "Notebook with impossible {pattern}",
        "Blueprint of non-existent structure",
        "Map to {liminal_place}",
        "Diagram drawn repeatedly",
        "Polaroid showing thirteen reflections",
    ],
    "LIMINAL_OBJECTS": [
        "Key that fits no door",
        "Ticket stub from event that never happened",
        "Receipt from {impossible_location}",
        "ID card with wrong photograph",
        "Coin from {alternate_year}",
    ],
    "PSYCHOGEOGRAPHIC": [
        "Stone from {significant_location}",
        "Graffiti symbol appearing everywhere",
        "Object found in wrong location",
        "Fragment of {anachronistic_technology}",
        "Souvenir from {mythical_place}",
    ],
}

# Initialize MCP server
server = Server("orange-mythos")

# In-memory corpus cache
_corpus_cache: Optional[dict] = None


def ensure_corpus_dir():
    """Ensure corpus directory exists"""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    if not CORPUS_FILE.exists():
        CORPUS_FILE.write_text(
            json.dumps(
                {
                    "stories": [],
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "version": "1.0",
                    },
                },
                indent=2,
            )
        )


def load_corpus() -> dict:
    """Load corpus from disk"""
    global _corpus_cache
    if _corpus_cache is None:
        ensure_corpus_dir()
        _corpus_cache = json.loads(CORPUS_FILE.read_text())
    return _corpus_cache


def save_corpus(corpus: dict):
    """Save corpus to disk"""
    global _corpus_cache
    ensure_corpus_dir()
    _corpus_cache = corpus
    CORPUS_FILE.write_text(json.dumps(corpus, indent=2))


def calculate_mythologization_score(story: dict) -> float:
    """
    Score a story's mythologization potential (0.0-1.0)

    Factors:
    - Ambiguity (unexplained elements)
    - Temporal liminality (dusk/dawn/midnight)
    - Youth involvement
    - Multiple witnesses
    - Symbolic resonance
    """
    score = 0.0
    text = story.get("text", "").lower()
    headline = story.get("headline", "").lower()

    # Ambiguity indicators
    ambiguity_terms = [
        "unexplained",
        "mysterious",
        "strange",
        "bizarre",
        "unknown",
        "unclear",
        "investigators puzzled",
    ]
    score += sum(0.1 for term in ambiguity_terms if term in text) * 0.15

    # Temporal liminality
    liminal_times = ["midnight", "dawn", "dusk", "3 am", "witching hour"]
    score += sum(0.15 for time in liminal_times if time in text) * 0.2

    # Youth involvement
    youth_terms = [
        "teen",
        "teenager",
        "youth",
        "student",
        "high school",
        "juvenile",
        "adolescent",
    ]
    if any(term in headline or term in text for term in youth_terms):
        score += 0.2

    # Multiple witnesses
    witness_terms = ["witnesses", "several people", "multiple", "group saw"]
    score += sum(0.1 for term in witness_terms if term in text) * 0.15

    # Symbolic resonance (already has object focus)
    object_terms = ["found", "discovered", "artifact", "object", "item"]
    score += sum(0.05 for term in object_terms if term in text) * 0.1

    # Location specificity (specific places = better mythology)
    if any(place in text for place in ["route ", "highway ", "avenue", "street"]):
        score += 0.1

    # Night occurrence
    if any(term in text for term in ["night", "evening", "after dark"]):
        score += 0.1

    return min(score, 1.0)


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available corpus resources"""
    ensure_corpus_dir()

    resources = [
        Resource(
            uri="orange://corpus/stats",
            name="Corpus Statistics",
            mimeType="application/json",
            description="Statistics about the mythologizable story corpus",
        )
    ]

    # Add resource for each story in corpus
    corpus = load_corpus()
    for story in corpus.get("stories", [])[:50]:  # Limit to 50 for performance
        story_id = story.get("id", "unknown")
        resources.append(
            Resource(
                uri=f"orange://story/{story_id}",
                name=f"Story: {story.get('headline', 'Untitled')}",
                mimeType="application/json",
                description=f"{story.get('date', 'Unknown date')} - {story.get('source', 'Unknown source')}",
            )
        )

    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a corpus resource"""
    corpus = load_corpus()

    if uri == "orange://corpus/stats":
        stories = corpus.get("stories", [])
        return json.dumps(
            {
                "total_stories": len(stories),
                "date_range": {
                    "earliest": min((s.get("date") for s in stories), default=None),
                    "latest": max((s.get("date") for s in stories), default=None),
                },
                "mythologized_count": sum(
                    1 for s in stories if s.get("mythologized", False)
                ),
                "avg_score": (
                    sum(s.get("mythologization_score", 0) for s in stories)
                    / len(stories)
                    if stories
                    else 0
                ),
                "top_tags": _count_tags(stories),
            },
            indent=2,
        )

    elif uri.startswith("orange://story/"):
        story_id = uri.split("/")[-1]
        story = next(
            (s for s in corpus.get("stories", []) if s.get("id") == story_id), None
        )
        if story:
            return json.dumps(story, indent=2)
        else:
            return json.dumps({"error": f"Story {story_id} not found"})

    return json.dumps({"error": f"Unknown resource: {uri}"})


def _count_tags(stories: list) -> dict:
    """Count tag occurrences"""
    tag_counts = {}
    for story in stories:
        for tag in story.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Orange Agent tools"""
    return [
        Tool(
            name="search_internet_archive_newspapers",
            description="""
            Search Internet Archive for New Jersey newspapers (1975-1995).
            Returns metadata about available newspaper issues.
            
            Parameters:
            - keywords: Search terms (e.g., "punk rock", "unexplained", "teenage crime")
            - date_range: Date range tuple (start_year, end_year)
            - max_results: Maximum results to return (default 20)
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search keywords"},
                    "start_year": {
                        "type": "integer",
                        "description": "Start year (1975-1995)",
                        "minimum": 1975,
                        "maximum": 1995,
                    },
                    "end_year": {
                        "type": "integer",
                        "description": "End year (1975-1995)",
                        "minimum": 1975,
                        "maximum": 1995,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20,
                    },
                },
                "required": ["keywords", "start_year", "end_year"],
            },
        ),
        Tool(
            name="add_story_to_corpus",
            description="""
            Add a mythologizable story to the corpus with automatic scoring.
            
            Parameters:
            - headline: Story headline
            - date: Publication date (YYYY-MM-DD)
            - source: Newspaper name
            - text: Full article text
            - location: Specific NJ location
            - tags: Category tags (e.g., ["rock_bands", "youth_crime"])
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
                    "source": {"type": "string"},
                    "text": {"type": "string"},
                    "location": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["headline", "date", "source", "text", "location", "tags"],
            },
        ),
        Tool(
            name="search_corpus",
            description="""
            Search the mythologizable story corpus.
            
            Parameters:
            - tags: Required tags (e.g., ["rock_bands", "youth_crime"])
            - min_score: Minimum mythologization score (0.0-1.0)
            - location: Specific NJ location
            - date_range: Date range tuple (start, end) in YYYY-MM-DD format
            - exclude_mythologized: Skip already mythologized stories (default true)
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required tags (must have at least 2)",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum mythologization score",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                    },
                    "location": {"type": "string", "description": "NJ location filter"},
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)",
                    },
                    "exclude_mythologized": {
                        "type": "boolean",
                        "description": "Exclude already mythologized stories",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="insert_symbolic_object",
            description="""
            Insert a symbolic object into a story for mythologization.
            
            The object should NOT exist in the original story but should be:
            - Contextually appropriate to time/place
            - Metaphorically resonant with themes
            - Mythologically significant
            
            Parameters:
            - story_id: ID of story to mythologize
            - object_category: One of: CIRCULAR_TIME, INFORMATION_ARTIFACTS, 
                               LIMINAL_OBJECTS, PSYCHOGEOGRAPHIC
            - custom_object: Optional custom object description
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "story_id": {"type": "string"},
                    "object_category": {
                        "type": "string",
                        "enum": [
                            "CIRCULAR_TIME",
                            "INFORMATION_ARTIFACTS",
                            "LIMINAL_OBJECTS",
                            "PSYCHOGEOGRAPHIC",
                        ],
                    },
                    "custom_object": {
                        "type": "string",
                        "description": "Custom object description (overrides category template)",
                    },
                },
                "required": ["story_id", "object_category"],
            },
        ),
        Tool(
            name="gonzo_rewrite",
            description="""
            Rewrite a mythologized story in gonzo journalism style.
            
            Style characteristics:
            - First-person embedded journalism
            - Paranoia and conspiracy undertones  
            - Perception shifts and altered states
            - Authority distrust
            - Vivid sensory details
            - Symbolic object becomes central
            
            Parameters:
            - story_id: ID of mythologized story
            - perspective: Narrative perspective (journalist, witness, investigator)
            - intensity: Gonzo intensity level (1-5, where 5 is full Hunter S. Thompson)
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "story_id": {"type": "string"},
                    "perspective": {
                        "type": "string",
                        "enum": [
                            "journalist",
                            "witness",
                            "investigator",
                            "participant",
                        ],
                        "default": "journalist",
                    },
                    "intensity": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                        "description": "Gonzo intensity (1=subtle, 5=full Thompson)",
                    },
                },
                "required": ["story_id"],
            },
        ),
        Tool(
            name="get_corpus_stats",
            description="""
            Get statistics about the mythologizable story corpus.
            
            Returns:
            - Total stories
            - Date range coverage
            - Tag distribution
            - Mythologization status
            - Average scores
            """,
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""

    if name == "search_internet_archive_newspapers":
        return await _search_internet_archive(arguments)

    elif name == "add_story_to_corpus":
        return await _add_story_to_corpus(arguments)

    elif name == "search_corpus":
        return await _search_corpus(arguments)

    elif name == "insert_symbolic_object":
        return await _insert_symbolic_object(arguments)

    elif name == "gonzo_rewrite":
        return await _gonzo_rewrite(arguments)

    elif name == "get_corpus_stats":
        return await _get_corpus_stats()

    raise ValueError(f"Unknown tool: {name}")


async def _search_internet_archive(args: dict) -> list[TextContent]:
    """Search Internet Archive for NJ newspapers"""
    if not IA_AVAILABLE:
        return [
            TextContent(
                type="text",
                text="Error: internetarchive library not installed. Run: pip install internetarchive",
            )
        ]

    keywords = args["keywords"]
    start_year = args["start_year"]
    end_year = args["end_year"]
    max_results = args.get("max_results", 20)

    # Build search query
    query = (
        f"collection:newspapers AND "
        f"subject:(New Jersey) AND "
        f"date:[{start_year}-01-01 TO {end_year}-12-31] AND "
        f"({keywords})"
    )

    try:
        results = []
        search_iter = search_items(query)

        for i, item in enumerate(search_iter):
            if i >= max_results:
                break

            results.append(
                {
                    "identifier": item.get("identifier"),
                    "title": item.get("title"),
                    "date": item.get("date"),
                    "description": item.get("description"),
                    "url": f"https://archive.org/details/{item.get('identifier')}",
                }
            )

        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"query": query, "results_count": len(results), "results": results},
                    indent=2,
                ),
            )
        ]

    except Exception as e:
        return [
            TextContent(type="text", text=f"Error searching Internet Archive: {str(e)}")
        ]


async def _add_story_to_corpus(args: dict) -> list[TextContent]:
    """Add a story to the corpus"""
    corpus = load_corpus()

    # Generate story ID
    story_id = f"nj_{args['date'].replace('-', '')}_{len(corpus['stories']):03d}"

    # Calculate mythologization score
    story = {
        "id": story_id,
        "headline": args["headline"],
        "date": args["date"],
        "source": args["source"],
        "text": args["text"],
        "location": args["location"],
        "tags": args["tags"],
        "mythologization_score": 0.0,
        "mythologized": False,
        "added": datetime.now().isoformat(),
    }

    story["mythologization_score"] = calculate_mythologization_score(story)

    corpus["stories"].append(story)
    save_corpus(corpus)

    return [
        TextContent(
            type="text",
            text=json.dumps(
                {
                    "success": True,
                    "story_id": story_id,
                    "mythologization_score": story["mythologization_score"],
                    "message": f"Added story '{args['headline']}' with score {story['mythologization_score']:.2f}",
                },
                indent=2,
            ),
        )
    ]


async def _search_corpus(args: dict) -> list[TextContent]:
    """Search the corpus for matching stories"""
    corpus = load_corpus()
    stories = corpus.get("stories", [])

    # Filter by tags
    if tags := args.get("tags"):
        stories = [s for s in stories if len(set(tags) & set(s.get("tags", []))) >= 2]

    # Filter by score
    if min_score := args.get("min_score"):
        stories = [s for s in stories if s.get("mythologization_score", 0) >= min_score]

    # Filter by location
    if location := args.get("location"):
        stories = [
            s for s in stories if location.lower() in s.get("location", "").lower()
        ]

    # Filter by date range
    if start_date := args.get("start_date"):
        stories = [s for s in stories if s.get("date", "") >= start_date]
    if end_date := args.get("end_date"):
        stories = [s for s in stories if s.get("date", "") <= end_date]

    # Filter mythologized
    if args.get("exclude_mythologized", True):
        stories = [s for s in stories if not s.get("mythologized", False)]

    # Sort by score
    stories.sort(key=lambda s: s.get("mythologization_score", 0), reverse=True)

    return [
        TextContent(
            type="text",
            text=json.dumps(
                {
                    "results_count": len(stories),
                    "results": stories[:20],  # Return top 20
                },
                indent=2,
            ),
        )
    ]


async def _insert_symbolic_object(args: dict) -> list[TextContent]:
    """Insert symbolic object into story"""
    corpus = load_corpus()
    story_id = args["story_id"]

    # Find story
    story = next((s for s in corpus["stories"] if s["id"] == story_id), None)
    if not story:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Story {story_id} not found"})
            )
        ]

    # Get or generate object
    if custom_obj := args.get("custom_object"):
        symbolic_object = custom_obj
    else:
        category = args["object_category"]
        templates = SYMBOLIC_OBJECTS.get(category, [])
        # For now, return template - Claude will contextualize it
        symbolic_object = templates[0] if templates else "Unknown object"

    # Add to story
    story["symbolic_object"] = {
        "category": args["object_category"],
        "description": symbolic_object,
        "inserted": datetime.now().isoformat(),
    }
    story["mythologized"] = True

    save_corpus(corpus)

    return [
        TextContent(
            type="text",
            text=json.dumps(
                {
                    "success": True,
                    "story_id": story_id,
                    "symbolic_object": story["symbolic_object"],
                    "message": f"Inserted {args['object_category']} object into story",
                },
                indent=2,
            ),
        )
    ]


async def _gonzo_rewrite(args: dict) -> list[TextContent]:
    """Generate gonzo rewrite prompt - actual rewrite done by Claude"""
    corpus = load_corpus()
    story_id = args["story_id"]

    story = next((s for s in corpus["stories"] if s["id"] == story_id), None)
    if not story:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Story {story_id} not found"})
            )
        ]

    if not story.get("mythologized"):
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": "Story must have symbolic object inserted first"}
                ),
            )
        ]

    perspective = args.get("perspective", "journalist")
    intensity = args.get("intensity", 3)

    # Build gonzo rewrite prompt for Claude
    prompt = f"""
Rewrite this story in gonzo journalism style:

**Original Story:**
Headline: {story['headline']}
Date: {story['date']}
Location: {story['location']}

{story['text']}

**Symbolic Object (INSERT INTO NARRATIVE):**
{story['symbolic_object']['description']}

**Style Parameters:**
- Perspective: {perspective}
- Intensity: {intensity}/5 (1=subtle, 5=full Hunter S. Thompson)

**Gonzo Requirements:**
- First-person embedded narrative
- Make the symbolic object CENTRAL to the story
- Paranoia and conspiracy undertones
- Authority distrust
- Vivid sensory details
- Perception shifts {"(hint at altered states)" if intensity >= 3 else ""}
{"- Multiple levels of reality/unreality" if intensity >= 4 else ""}
{"- Full gonzo madness - American Dream style" if intensity == 5 else ""}

The object should feel like it was ALWAYS part of the story, but also impossibly significant.
"""

    return [TextContent(type="text", text=prompt)]


async def _get_corpus_stats() -> list[TextContent]:
    """Get corpus statistics"""
    corpus = load_corpus()
    stories = corpus.get("stories", [])

    if not stories:
        return [
            TextContent(
                type="text",
                text=json.dumps({"total_stories": 0, "message": "Corpus is empty"}),
            )
        ]

    stats = {
        "total_stories": len(stories),
        "date_range": {
            "earliest": min((s.get("date") for s in stories), default=None),
            "latest": max((s.get("date") for s in stories), default=None),
        },
        "mythologized_count": sum(1 for s in stories if s.get("mythologized", False)),
        "unmythologized_count": sum(
            1 for s in stories if not s.get("mythologized", False)
        ),
        "avg_mythologization_score": sum(
            s.get("mythologization_score", 0) for s in stories
        )
        / len(stories),
        "top_tags": _count_tags(stories),
        "top_locations": _count_locations(stories),
        "score_distribution": {
            "high (>0.7)": sum(
                1 for s in stories if s.get("mythologization_score", 0) > 0.7
            ),
            "medium (0.4-0.7)": sum(
                1 for s in stories if 0.4 <= s.get("mythologization_score", 0) <= 0.7
            ),
            "low (<0.4)": sum(
                1 for s in stories if s.get("mythologization_score", 0) < 0.4
            ),
        },
    }

    return [TextContent(type="text", text=json.dumps(stats, indent=2))]


def _count_locations(stories: list) -> dict:
    """Count location occurrences"""
    loc_counts = {}
    for story in stories:
        loc = story.get("location", "Unknown")
        loc_counts[loc] = loc_counts.get(loc, 0) + 1
    return dict(sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)[:10])


async def main():
    """Run the MCP server"""
    ensure_corpus_dir()

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
