import pytest
from pathlib import Path
from typing import Any, Iterable, cast
from PIL import Image
from app.agents.tools.image_tools import (
    composite_images,
    composite_character_portrait,
)


def _data_list(img: Image.Image) -> list:
    """Return the pixel data as a list with a typing-cast for ImagingCore."""
    return list(cast(Iterable[Any], cast(object, img.getdata())))


def _make_rgba(path: Path, size=(10, 10), color=(255, 0, 0, 255)):
    """Create and save a simple RGBA PNG for tests."""
    img = Image.new("RGBA", size, color)
    img.save(path, format="PNG")
    img.close()
    return path


def _pil_composite_from_paths(paths):
    """Return a PIL Image that alpha-composites the given image paths in order."""
    imgs = [Image.open(p).convert("RGBA") for p in paths]
    base = imgs[0]
    for layer in imgs[1:]:
        if layer.size != base.size:
            layer = layer.resize(base.size, Image.Resampling.LANCZOS)
        base = Image.alpha_composite(base, layer)
    for im in imgs:
        try:
            im.close()
        except ValueError:
            pass
    return base


def test_composite_images_basic(tmp_path):
    base = tmp_path / "base.png"
    layer = tmp_path / "layer.png"
    _make_rgba(base, size=(16, 16), color=(200, 50, 50, 255))
    _make_rgba(layer, size=(16, 16), color=(50, 50, 200, 128))  # semi-transparent blue

    out = tmp_path / "out.png"
    ret = composite_images(out, [base, layer])

    assert isinstance(ret, str)
    out_path = Path(ret)
    assert out_path.exists()

    expected = _pil_composite_from_paths([base, layer])
    result = Image.open(out_path).convert("RGBA")

    assert _data_list(result) == _data_list(expected)


def test_composite_images_resizes_layers(tmp_path):
    base = tmp_path / "base2.png"
    small_layer = tmp_path / "small.png"
    _make_rgba(base, size=(20, 20), color=(10, 120, 10, 255))
    _make_rgba(
        small_layer, size=(5, 5), color=(200, 200, 0, 200)
    )  # smaller semi-opaque

    out = tmp_path / "resized_out.png"
    composite_images(out, [base, small_layer])

    expected = _pil_composite_from_paths([base, small_layer])
    result = Image.open(out).convert("RGBA")
    assert result.size == expected.size
    assert _data_list(result) == _data_list(expected)


def test_composite_character_portrait_wrapper(tmp_path):
    base = tmp_path / "char_base.png"
    trait1 = tmp_path / "eyes.png"
    trait2 = tmp_path / "hat.png"

    _make_rgba(base, size=(12, 12), color=(100, 100, 255, 255))
    _make_rgba(trait1, size=(12, 12), color=(0, 255, 0, 180))
    _make_rgba(trait2, size=(12, 12), color=(255, 255, 0, 128))

    out = tmp_path / "portrait" / "char.png"
    ret = composite_character_portrait(base, [trait1, trait2], out)

    assert isinstance(ret, str)
    out_path = Path(ret)
    assert out_path.exists()

    expected = _pil_composite_from_paths([base, trait1, trait2])
    result = Image.open(out_path).convert("RGBA")
    assert _data_list(result) == _data_list(expected)


def test_output_dir_created_and_return_value(tmp_path):
    base = tmp_path / "b.png"
    _make_rgba(base)

    nested = tmp_path / "nested" / "dir" / "result.png"
    assert not nested.parent.exists()
    ret = composite_images(nested, [base])
    assert nested.parent.exists()
    assert Path(ret).exists()
    assert str(nested) == ret or Path(ret).samefile(nested)


def test_no_images_raises_value_error(tmp_path):
    out = tmp_path / "no.png"
    with pytest.raises(ValueError):
        composite_images(out, [])
