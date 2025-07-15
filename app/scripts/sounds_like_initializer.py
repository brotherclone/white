import asyncio

from app.objects.db_models.artist_schema import RainbowArtist
from app.resources.plans.negative_artist_reference import NEGATIVE_ARTISTS
from app.resources.plans.positive_artist_reference import POSITIVE_ARTISTS
from app.scripts.reference_plan_helper import enrich_sounds_like, db_arist_to_rainbow_artist


async def initialize_sounds_like(artist_list):
    for artist_name in artist_list:

        an_artist = RainbowArtist(
            name=artist_name,
            id=None,
            discogs_id=None,
            profile=None,
        )
        try:
            a_rainbow_artist: RainbowArtist = await enrich_sounds_like(an_artist)
            if a_rainbow_artist:
                print(
                    f"Enriched sounds like for {a_rainbow_artist.name}: {a_rainbow_artist.profile if a_rainbow_artist.profile else 'No profile available'}")
            else:
                print(f"No sounds like found for {artist_name}")
        except Exception as e:
            print(f"Error enriching sounds like for {artist_name}: {e}")


if __name__ == "__main__":
    async def main():
        await initialize_sounds_like(NEGATIVE_ARTISTS)
        await initialize_sounds_like(POSITIVE_ARTISTS)


    asyncio.run(main())
