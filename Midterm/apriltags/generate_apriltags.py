"""Generate a printable letter-size (8.5 x 11 in) sheet of AprilTags (tag36h11).

Usage: python generate_apriltags.py [count] [first_id]
Writes apriltags_<first>-<last>.pdf/.png next to this script. Print at 100% / "Actual size".
"""
import os
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont

DPI = 300
PAGE_W, PAGE_H = int(8.5 * DPI), int(11 * DPI)
COLS, ROWS = 3, 4
TAG_IN = 2.0          # printed edge length of the tag's outer black square, inches
CELL_W, CELL_H = 2.5, 2.5  # grid cell size, inches


def make_tag(tag_id, px):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # 36h11 is 8 cells across including the black border; render at an exact multiple
    img = cv2.aruco.generateImageMarker(d, tag_id, 8 * 100)
    return Image.fromarray(img).resize((px, px), Image.NEAREST)


def main(count=12, first_id=0):
    page = Image.new("L", (PAGE_W, PAGE_H), 255)
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()

    tag_px = int(TAG_IN * DPI)
    x0 = (PAGE_W - COLS * CELL_W * DPI) / 2
    y0 = (PAGE_H - ROWS * CELL_H * DPI) / 2
    for i in range(count):
        r, c = divmod(i, COLS)
        cx = x0 + (c + 0.5) * CELL_W * DPI
        cy = y0 + (r + 0.5) * CELL_H * DPI
        tx, ty = int(cx - tag_px / 2), int(cy - tag_px / 2 - 20)
        page.paste(make_tag(first_id + i, tag_px), (tx, ty))
        label = f"ID {first_id + i}  (36h11, {TAG_IN:g} in)"
        w = draw.textlength(label, font=font)
        draw.text((cx - w / 2, ty + tag_px + 15), label, fill=0, font=font)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"apriltags_{first_id}-{first_id + count - 1}")
    page.save(out + ".pdf", resolution=DPI)
    page.save(out + ".png", dpi=(DPI, DPI))
    print("wrote", out + ".pdf")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
