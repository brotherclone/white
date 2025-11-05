from app.structures.music.rainbow_table.rainbow_table_song import \
    RainbowTableSong


def test_rainbow_table_song():
    """Test RainbowTableSong creation with minimal fields"""
    song = RainbowTableSong()
    assert song.bpm is None
    assert song.tempo is None
    assert song.key is None
    assert song.total_running_time is None
    assert song.album_id is None
    assert song.sequence_on_album is None


def test_rainbow_table_song_with_all_fields():
    """Test RainbowTableSong with all fields populated"""
    song = RainbowTableSong(
        bpm=120,
        total_running_time=180000,  # 3 minutes in milliseconds
        album_id="album-123",
        sequence_on_album=5,
    )
    assert song.bpm == 120
    assert song.total_running_time == 180000
    assert song.album_id == "album-123"
    assert song.sequence_on_album == 5


def test_rainbow_table_song_with_album_id_types():
    """Test that album_id accepts string or int"""
    song1 = RainbowTableSong(album_id=123)
    assert song1.album_id == 123

    song2 = RainbowTableSong(album_id="ABC-456")
    assert song2.album_id == "ABC-456"


def test_rainbow_table_song_with_sequence():
    """Test sequence_on_album field"""
    song = RainbowTableSong(sequence_on_album=1)
    assert song.sequence_on_album == 1
