from enum import Enum


class InfranymVoiceProfile(str, Enum):
    """Voice personality presets for different layers/moods"""

    ROBOTIC = "robotic"  # Flat, mechanical, alien
    WHISPER = "whisper"  # Breathy, intimate, mysterious
    PROCLAMATION = "proclaim"  # Clear, authoritative, ritual
    DISTORTED = "distorted"  # Glitchy, broken, corrupted
    ANCIENT = "ancient"  # Slow, deep, timeless
