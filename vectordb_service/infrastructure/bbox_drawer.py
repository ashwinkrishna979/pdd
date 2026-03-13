"""Draw highlight regions on frame images."""

import os
from PIL import Image, ImageDraw


def draw_highlight(
    image_path: str,
    region: list[float],
    highlight_type: str,
    output_dir: str,
    color: tuple[int, int, int] = (255, 80, 80),
    opacity: int = 50,
    outline_width: int = 3,
) -> str:
    """Draw a semi-transparent highlight on the image and save annotated copy.

    Args:
        image_path: Path to the source frame image.
        region: Normalised [x1, y1, x2, y2] coordinates (0.0–1.0).
        highlight_type: One of 'circle', 'square', or 'none'.
        output_dir: Directory to save the annotated image.
        color: RGB colour for the highlight.
        opacity: Fill opacity (0 = fully transparent, 255 = fully opaque).
        outline_width: Outline width in pixels.

    Returns:
        Path to the saved annotated image.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    if highlight_type == "none":
        # No highlight — just save a copy
        out_img = img.convert("RGB")
    else:
        x1 = region[0] * w
        y1 = region[1] * h
        x2 = region[2] * w
        y2 = region[3] * h

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if highlight_type == "circle":
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            rx = (x2 - x1) / 2
            ry = (y2 - y1) / 2
            radius = max(rx, ry) * 1.15
            shape_bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(shape_bbox, fill=(*color, opacity), outline=(*color, 200), width=outline_width)
        else:
            # square / rectangle
            draw.rectangle([x1, y1, x2, y2], fill=(*color, opacity), outline=(*color, 200), width=outline_width)

        out_img = Image.alpha_composite(img, overlay).convert("RGB")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_annotated.png")
    out_img.save(out_path)
    return out_path
