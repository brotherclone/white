import os
import discogs_client
from dotenv import load_dotenv

def get_artist(artist_id):
    try:
        artist = discogs.artist(artist_id)
        return artist
    except Exception as e:
        print(f"Error fetching artist with ID {artist_id}: {e}")
        return None

def search_artist(artist_name):
    """
    Search for an artist by name
    """
    try:
        results = discogs.search(artist_name, type='artist')
        if not results:
            print(f"No results found for {artist_name}")
            return None
        return results[0]
    except Exception as e:
        print(f"Error searching for artist {artist_name}: {e}")
        return None

def get_group_members(group_name):
    results = discogs.search(group_name, type='artist')
    if not results:
        print(f"No results found for {group_name}")
        return []
    group = results[0]
    group_detail = discogs.artist(group.id)
    members = []
    for m in group_detail.members:
        members.append({
            'id': m.id,
            'name': m.name
        })

    return members


def get_release_list(artist_id, per_page=50, page=1, sort='year', sort_order='asc',
                     release_type=None, main_release=True, debug=False):
    """
    Get releases for an artist with filtering options
    """
    try:
        artist = discogs.artist(artist_id)
        artist.releases.per_page = per_page
        artist.releases.sort = sort
        artist.releases.sort_order = sort_order
        artist_releases = artist.releases.page(page)
        if debug and len(artist_releases) > 0:
            print("First release attributes:", dir(artist_releases[0]))
            print("First release data:", artist_releases[0].__dict__)
        filtered_releases = []
        for release in artist_releases:
            if main_release and hasattr(release, 'role'):
                if debug:
                    print(f"Release: {release.title}, Role: {release.role}")
                if release.role not in ["Main", "Main Artist", "Primary Artist"]:
                    continue
            if release_type and hasattr(release, 'type') and release.type != release_type:
                continue
            filtered_releases.append(release)
        return filtered_releases

    except Exception as e:
        print(f"Error fetching releases for artist with ID {artist_id}: {e}")
        return []


if __name__ == "__main__":
    load_dotenv()
    discogs = discogs_client.Client(
        'RainbowProfiler/1.0',
        user_token=os.environ['USER_ACCESS_TOKEN'],
    )
    a = search_artist("David Bowie")
    print(a.id)
    b = get_artist(a.id)
    print(b.profile)
