from pathlib import Path
from typing import List, Union
from PIL import Image


def composite_images(
    output_path: Union[str, Path], image_paths: List[Union[str, Path]]
) -> str:
    """Composite multiple PNG layers into a single image.

    Stacks images in order with proper alpha blending. First image
    is the base layer, subsequent images are overlaid on top.

    Args:
        output_path: Where to save the composite image
        image_paths: List of PNG paths to stack (order matters)

    Returns:
        String path to the saved composite image

    Example:
        >>> composite_images(
        ...     "character.png",
        ...     ["base.png", "eyes.png", "clothes.png"]
        ... )
    """
    if not image_paths:
        raise ValueError("Must provide at least one image path")
    output_path = Path(output_path)
    image_paths = [Path(p) for p in image_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite = Image.open(image_paths[0]).convert("RGBA")
    for layer_path in image_paths[1:]:
        layer = Image.open(layer_path).convert("RGBA")
        if layer.size != composite.size:
            layer = layer.resize(composite.size, Image.Resampling.LANCZOS)
        composite = Image.alpha_composite(composite, layer)
    composite.save(output_path, "PNG")

    return str(output_path)


def composite_character_portrait(
    base_layer: Union[str, Path],
    trait_layers: List[Union[str, Path]],
    output_path: Union[str, Path],
) -> str:
    """Helper specifically for character portraits.

    Convenience wrapper around composite_images that makes the
    base + traits pattern more explicit.

    Args:
        base_layer: Base character image
        trait_layers: Overlay traits (eyes, clothes, accessories, etc.)
        output_path: Where to save result

    Returns:
        String path to saved composite
    """
    all_layers = [base_layer] + trait_layers
    return composite_images(output_path, all_layers)
