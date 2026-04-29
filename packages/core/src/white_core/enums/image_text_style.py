import enum


class ImageTextStyle(str, enum.Enum):
    CLEAN = "clean"
    DEFAULT = "default"
    GLITCH = "glitch"
    STATIC = "static"
