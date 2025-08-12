import httpx
import logging
import sys

from typing import Any
from mcp.server.fastmcp import FastMCP

USER_AGENT = "earthly_frames/1.0"
TIME_OUT = 10.0
BASE_URL = "https://www.earthlyframes.com"

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("earthly_frames")

async def earthly_frames_retriever_service(url:str)-> dict[str, Any] | None :
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            response = await client.get(url, headers=headers, timeout=TIME_OUT)
            return response.json()
        except Exception as e:
            logging.info(e)
            return None

@mcp.tool()
async def get_all_recordings(filter_for_rainbow_table: bool):
    endpoint = f"{BASE_URL}/albums.json"
    recording_data = await earthly_frames_retriever_service(endpoint)
    if not recording_data:
        return "Unable to retrieve recordings"
    if filter_for_rainbow_table:
        return [
            record for record in recording_data
            if record.get('rainbow_table') != "not_associated"
        ]
    return recording_data

@mcp.tool()
async def album_for_color(color: str):
    endpoint = f"{BASE_URL}/albums.json"
    album_data = await earthly_frames_retriever_service(endpoint)
    if not album_data:
        return "Unable to retrieve albums"
    return [
        album for album in album_data
        if album.get('rainbow_table') and album['rainbow_table'].lower() == color.lower()
    ]

@mcp.tool()
async def get_album(album_id: str | int):
    endpoint = f"{BASE_URL}/albums/{album_id}.json"
    album_data = await earthly_frames_retriever_service(endpoint)
    if not album_data:
        return "Unable to retrieve album"
    return album_data

@mcp.tool()
async def get_song(album_id: str | int, song_id: str | int):
    endpoint = f"{BASE_URL}/albums/{album_id}/songs/{song_id}.json"
    song_data = await earthly_frames_retriever_service(endpoint)
    if not song_data:
        return "Unable to retrieve song"
    return song_data

mcp.tools = [
    get_all_recordings,
    album_for_color,
    get_album,
    get_song
]


if __name__ == "__main__":
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        raise
