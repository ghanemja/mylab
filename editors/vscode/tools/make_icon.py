"""Draw the extension icon (images/icon.png): a teal rounded square with a white wave.

    python editors/vscode/tools/make_icon.py

Needs Pillow. The drawing is original to this project.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
SCALE = 4  # draw large, then shrink for smooth edges


def main() -> None:
    n = SIZE * SCALE
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, n - 1, n - 1], radius=n * 0.2, fill=(13, 148, 136, 255))
    # One and a half periods of a sine wave, drawn as a thick round-capped line.
    left, right, mid, amp, w = n * 0.16, n * 0.84, n * 0.5, n * 0.22, n * 0.075
    pts = []
    for k in range(3001):
        t = k / 3000
        pts.append((left + t * (right - left), mid - amp * math.sin(t * 3 * math.pi)))
    for x, y in pts:  # stamping circles gives a smooth stroke with round ends
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill="white")
    out = Path(__file__).resolve().parent.parent / "images" / "icon.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize((SIZE, SIZE), Image.LANCZOS).save(out, optimize=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
