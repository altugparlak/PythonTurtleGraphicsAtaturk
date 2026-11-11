# PythonTurtleGraphicsAtaturk

This python code is written with the hard-coding method. So, another drawing cannot be created by simply changing the code.


<img width="905" alt="Ekran Resmi 2023-03-26 12 01 38" src="https://user-images.githubusercontent.com/57351922/227765755-33b447d5-4100-43d1-af09-31da30c9cf95.png">

## Draw your own picture

`draw.py` draws any picture in the same style, automatically:

1. It finds the face with a Haar cascade and crops to head and shoulders (if there is no face, it uses the whole picture).
2. It turns the picture into black and white with Otsu's thresholding.
3. It scans the result row by row and draws every black run with the turtle.

```bash
pip install -r requirements.txt
```

```bash
python draw.py photo.jpg
```

Run it without a file name to choose a picture from a file dialog.

| Option | What it does |
| --- | --- |
| `--no-face` | Skip face detection and draw the whole picture (for objects) |
| `--invert` | Draw the light areas instead of the dark ones |
| `--margin 0.8` | How much to keep around the face, in face sizes |
| `--size 600` | Drawing size in pixels |
| `--step 4` | Row spacing and pen size; smaller means more detail |
| `--fast` | Show the finished drawing at once |
| `--preview out.png` | Save the result to a file instead of opening a window |

Otsu uses a single threshold, so pictures with a plain background work best.
OpenCV 5 no longer ships the Haar cascade, which is why `requirements.txt` pins OpenCV 4.
