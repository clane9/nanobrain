import argparse
import base64
import fnmatch
import io
import random
from threading import Thread
from queue import Queue

import fsspec
from flask import Flask, jsonify, render_template_string
from PIL import Image, ImageDraw

GRID_CELL_WIDTH = 128  # px min-width of grid cells
IMAGE_WIDTH = 224
JPEG_QUALITY = 85
BATCH_SIZE = 16

VIEWS = ("ax", "cor", "sag")

app = Flask(__name__)

_queue: Queue = Queue(maxsize=2048)

HTML = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>sMRI viewer</title>
  <style>
    body {{ margin: 0; background: #111; }}
    #grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax({GRID_CELL_WIDTH}px, 1fr));
      gap: 4px;
      padding: 4px;
    }}
    img {{ width: 100%; aspect-ratio: 1 / 1; display: block; }}
    #sentinel {{ height: 1px; }}
  </style>
</head>
<body>
  <div id="grid"></div>
  <div id="sentinel"></div>
  <script>
    const PREFETCH = 3;
    const grid = document.getElementById('grid');
    const sentinel = document.getElementById('sentinel');
    const queue = [];
    let loading = false;
    let done = false;

    function enqueue() {{
      while (!done && queue.length < PREFETCH) {{
        queue.push(fetch('/batch').then(r => r.json()));
      }}
    }}

    const visible = () => sentinel.getBoundingClientRect().top < window.innerHeight;

    async function loadBatch() {{
      if (loading || done) return;
      loading = true;
      while (!done && visible()) {{
        enqueue();
        const images = await queue.shift();
        if (images.length === 0) {{
          done = true;
          observer.disconnect();
          break;
        }}
        for (const b64 of images) {{
          const img = document.createElement('img');
          img.src = 'data:image/jpeg;base64,' + b64;
          grid.appendChild(img);
        }}
      }}
      loading = false;
    }}

    const observer = new IntersectionObserver(entries => {{
      if (entries[0].isIntersecting) loadBatch();
    }});

    observer.observe(sentinel);
  </script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/batch")
def batch():
    images = []
    while len(images) < BATCH_SIZE:
        img: Image.Image | None = _queue.get()
        if img is None:  # sentinel: stream exhausted
            _queue.put(None)  # keep it for subsequent requests
            break
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=JPEG_QUALITY)
        images.append(base64.b64encode(buf.getvalue()).decode())
    return jsonify(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/FOMO300K_images")
    parser.add_argument("--filelist", type=str, default=None)
    parser.add_argument("--view", choices=VIEWS, default="ax", help="view (default: ax)")
    parser.add_argument("--port", type=int, default=5023)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fs, root = fsspec.url_to_fs(args.root)
    filelist = args.filelist or f"{args.root}/filelist_subsampled.txt"
    with fsspec.open(filelist, "rt") as f:
        paths = f.read().strip().splitlines()
    paths = [p.replace(".npz", f".{args.view}.jpg") for p in paths]

    if args.pattern:
        paths = fnmatch.filter(paths, args.pattern)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(paths)
    print(f"found {len(paths)} thumbnails")

    def fn():
        for path in paths:
            with fs.open(f"{root}/{path}", "rb") as f:
                square = Image.open(f)
                square.load()
            suffix = path.removesuffix(f".{args.view}.jpg").split("_")[-1]
            label = f"{path[:32]} {suffix}"
            ImageDraw.Draw(square).text((3, 2), label, fill=255)
            _queue.put(square)
        _queue.put(None)  # sentinel

    thread = Thread(target=fn, daemon=True)
    thread.start()

    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
