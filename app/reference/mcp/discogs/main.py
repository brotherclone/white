import logging
import os
import sys
from typing import Any

import discogs_client
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

USER_AGENT = "earthly_frames_discogs/1.0"
TIME_OUT = 30.0

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("earthly_frames_discogs")


async def discogs_earthly_frames_retriever_service() -> Any:
    load_dotenv()
    discogs = discogs_client.Client(
        USER_AGENT,
        user_token=os.environ["USER_ACCESS_TOKEN"],
    )
    return discogs


@mcp.tool()
async def look_up_artist_by_name(artist_name: str):
    try:
        discogs = await discogs_earthly_frames_retriever_service()
        results = discogs.search(artist_name, type="artist")
        if not results:
            logging.error(f"No results found for artist: {artist_name}")
            return None
        artist = results[0]
        result = {
            "id": artist.id,
            "name": artist.name,
            "type": getattr(artist, "type", None),
            "url": getattr(artist, "url", None),
        }
        print(f"Converted artist: {result}", file=sys.stderr)
        return result
    except Exception as err:
        logging.info(f"Error searching for artist {artist_name}: {err}")
        return None


@mcp.tool()
async def look_up_artist_by_id(artist_id: str | int):
    try:
        discogs = await discogs_earthly_frames_retriever_service()
        artist = discogs.artist(artist_id)
        return {
            "id": artist.id,
            "name": artist.name,
            "real_name": getattr(artist, "real_name", None),
            "profile": getattr(artist, "profile", None),
            "url": getattr(artist, "url", None),
            "genres": getattr(artist, "genres", []),
            "styles": getattr(artist, "styles", []),
        }
    except Exception as err:
        logging.info(f"Error fetching artist with ID {artist_id}: {err}")
        return None


@mcp.tool()
async def get_group_members(group_name: str):
    try:
        discogs = await discogs_earthly_frames_retriever_service()
        results = discogs.search(group_name, type="artist")
        if not results:
            logging.error(f"No results found for group: {group_name}")
            return []
        group = results[0]
        group_detail = discogs.artist(group.id)
        members = [{"id": m.id, "name": m.name} for m in group_detail.members]
        return members
    except Exception as err:
        logging.info(f"Error fetching group members for {group_name}: {err}")
        return []


@mcp.tool()
async def get_release_list(
    artist_id,
    per_page=50,
    page=1,
    sort="year",
    sort_order="asc",
    release_type=None,
    main_release=True,
):
    try:
        discogs = await discogs_earthly_frames_retriever_service()
        artist = discogs.artist(artist_id)
        artist.releases.per_page = per_page
        artist.releases.sort = sort
        artist.releases.sort_order = sort_order
        artist_releases = artist.releases.page(page)
        filtered_releases = []
        for release in artist_releases:
            if main_release and hasattr(release, "role"):
                if release.role not in ["Main", "Main Artist", "Primary Artist"]:
                    continue
            if (
                release_type
                and hasattr(release, "type")
                and release.type != release_type
            ):
                continue
            filtered_releases.append(
                {
                    "id": release.id,
                    "title": release.title,
                    "year": getattr(release, "year", None),
                    "type": getattr(release, "type", None),
                    "role": getattr(release, "role", None),
                    "url": getattr(release, "url", None),
                }
            )
        return filtered_releases
    except Exception as err:
        logging.info(f"Error fetching releases for artist with ID {artist_id}: {err}")
        return []


mcp.tools = [
    look_up_artist_by_name,
    look_up_artist_by_id,
    get_group_members,
    get_release_list,
]

if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        raise
