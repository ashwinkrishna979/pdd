"""Draw highlight regions on frame images."""

import io
import os
from PIL import Image, ImageDraw


def _draw_highlight_on_image(
    img: Image.Image,
    region: list[float],
    highlight_type: str,
    color: tuple[int, int, int] = (255, 80, 80),
    opacity: int = 50,
    outline_width: int = 3,
) -> Image.Image:
    """Core highlight drawing logic. Returns an RGB PIL Image."""
    img = img.convert("RGBA")
    w, h = img.size

    if highlight_type == "none":
        return img.convert("RGB")

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
        draw.rectangle([x1, y1, x2, y2], fill=(*color, opacity), outline=(*color, 200), width=outline_width)

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_highlight(
    image_path: str,
    region: list[float],
    highlight_type: str,
    output_dir: str,
    color: tuple[int, int, int] = (255, 80, 80),
    opacity: int = 50,
    outline_width: int = 3,
) -> str:
    """Draw a semi-transparent highlight on the image and save annotated copy to disk.

    Returns:
        Path to the saved annotated image.
    """
    img = Image.open(image_path)
    out_img = _draw_highlight_on_image(img, region, highlight_type, color, opacity, outline_width)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_annotated.png")
    out_img.save(out_path)
    return out_path


def draw_highlight_to_bytes(
    image_path: str,
    region: list[float],
    highlight_type: str,
    color: tuple[int, int, int] = (255, 80, 80),
    opacity: int = 50,
    outline_width: int = 3,
) -> bytes:
    """Draw a semi-transparent highlight and return the result as PNG bytes.

    Used when storing annotated images in MongoDB instead of filesystem.
    """
    img = Image.open(image_path)
    out_img = _draw_highlight_on_image(img, region, highlight_type, color, opacity, outline_width)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()
