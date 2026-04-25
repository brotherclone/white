import json

from white_ideation.reference.mcp.rows_bud.orange_corpus import get_corpus

with open(
    "/Volumes/LucidNonsense/White/app/reference/mcp/rows_bud/mythologizable_corpus.json",
    "r",
) as f:
    data = json.load(f)

corpus = get_corpus()
print(f"📚 Starting: {len(corpus.df)} stories")

for story in data["stories"]:
    corpus.add_story(
        headline=story["headline"],
        date=story["date"],
        source=story["source"],
        text=story["text"],
        location=story["location"],
        tags=story["tags"],
    )
    print(f"   ✓ {story['headline'][:50]}...")

print(f"✅ Final: {len(corpus.df)} stories")
