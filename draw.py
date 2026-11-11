"""Draw any picture with turtle graphics, in the style of turtleMustafaKemal.py.

Pipeline:
  1. Find the face (Haar cascade) and crop to head and shoulders.
  2. Turn the crop into black and white with Otsu's thresholding.
  3. Scan it row by row and draw every black run as a horizontal pen stroke.

Usage:
  python draw.py photo.jpg
  python draw.py                      # opens a file picker
  python draw.py photo.jpg --preview out.png   # no window, just save the result
"""

import argparse
import sys

import cv2
import numpy as np


def detect_face(gray):
    """Return the largest face as (x, y, w, h), or None if no face is found."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=8, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def crop_around_face(gray, face, margin):
    """Crop to the face plus `margin` face-sizes on each side (more below)."""
    x, y, w, h = face
    height, width = gray.shape
    left = max(0, int(x - margin * w))
    right = min(width, int(x + w + margin * w))
    top = max(0, int(y - margin * h))
    bottom = min(height, int(y + h + 1.5 * margin * h))
    return gray[top:bottom, left:right]


def binarize(gray, invert=False):
    """Otsu's thresholding. Returns (mask, threshold); mask is True where ink goes."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = binary == 0
    if invert:
        mask = ~mask
    return mask, threshold


def to_segments(mask, size, step):
    """Convert an ink mask into horizontal strokes on a size x size canvas.

    Returns a list of rows, top to bottom; each row is (y, [(x_start, x_end), ...])
    in turtle coordinates (origin in the centre, y pointing up).
    """
    height, width = mask.shape
    scale = size / max(height, width)
    cols = max(1, round(width * scale))
    rows = max(1, round(height * scale / step))
    small = cv2.resize(
        mask.astype(np.uint8) * 255, (cols, rows), interpolation=cv2.INTER_AREA
    )
    ink = small > 127

    result = []
    for r in range(rows):
        # Pad with False so every run has a clear start and end.
        line = np.concatenate(([False], ink[r], [False]))
        changes = np.flatnonzero(line[1:] != line[:-1])
        runs = [
            (int(start) - cols // 2, int(end) - cols // 2)
            for start, end in zip(changes[::2], changes[1::2])
        ]
        if runs:
            result.append((rows * step // 2 - r * step, runs))
    return result


def process(path, size=600, step=4, margin=0.8, use_face=True, invert=False):
    """Run the whole image pipeline. Returns (rows, info dict)."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face = detect_face(gray) if use_face else None
    if face is not None:
        gray = crop_around_face(gray, face, margin)

    mask, threshold = binarize(gray, invert)
    rows = to_segments(mask, size, step)
    return rows, {"face": face, "threshold": threshold}


def save_preview(rows, path, size, step):
    """Render the strokes to an image file, without opening a turtle window."""
    canvas = np.full((size + 2 * step, size + 2 * step), 255, np.uint8)
    centre = canvas.shape[0] // 2
    for y, runs in rows:
        for x_start, x_end in runs:
            cv2.line(
                canvas, (centre + x_start, centre - y), (centre + x_end, centre - y),
                0, step,
            )
    cv2.imwrite(path, canvas)


def draw(rows, size, step, fast=False):
    """Draw the strokes with turtle, one row at a time from the top."""
    import turtle

    screen = turtle.Screen()
    screen.setup(size + 100, size + 100)
    screen.bgcolor("white")
    screen.tracer(0)

    pen = turtle.Turtle()
    pen.color("black")
    pen.pensize(step)
    pen.hideturtle()
    pen.up()

    for y, runs in rows:
        for x_start, x_end in runs:
            pen.setpos(x_start, y)
            pen.down()
            pen.setpos(x_end, y)
            pen.up()
        if not fast:
            screen.update()  # show the picture growing row by row
    screen.update()
    turtle.done()


def pick_file():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose a picture",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*")],
    )
    root.destroy()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image", nargs="?", help="picture to draw (file picker if omitted)")
    parser.add_argument("--size", type=int, default=600, help="drawing size in pixels (default 600)")
    parser.add_argument("--step", type=int, default=4, help="row spacing and pen size (default 4)")
    parser.add_argument("--margin", type=float, default=0.8,
                        help="how much to keep around the face, in face sizes (default 0.8)")
    parser.add_argument("--no-face", action="store_true", help="skip face detection, draw the whole picture")
    parser.add_argument("--invert", action="store_true", help="draw the light areas instead of the dark ones")
    parser.add_argument("--fast", action="store_true", help="show the finished drawing at once")
    parser.add_argument("--preview", metavar="FILE", help="save the result to an image file instead of drawing")
    args = parser.parse_args()

    path = args.image or pick_file()
    if not path:
        sys.exit("No picture chosen.")

    try:
        rows, info = process(
            path, args.size, args.step, args.margin, not args.no_face, args.invert
        )
    except ValueError as error:
        sys.exit(str(error))

    if args.no_face:
        print("Face detection skipped, drawing the whole picture.")
    elif info["face"] is None:
        print("No face found, drawing the whole picture.")
    else:
        print("Face found at x=%d y=%d w=%d h=%d." % info["face"])
    print(f"Otsu threshold: {info['threshold']:.0f}")

    if args.preview:
        save_preview(rows, args.preview, args.size, args.step)
        print(f"Saved {args.preview}")
    else:
        draw(rows, args.size, args.step, args.fast)


if __name__ == "__main__":
    main()
