from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Polarisoo/eecs498-598-dlcv-wi2022.git"
COLAB_REPO = "/content/eecs498-598-dlcv-wi2022"


NOTEBOOKS = [
    ("assignments/a4-object-detection/one_stage_detector.ipynb", "assignments/a4-object-detection"),
    ("assignments/a4-object-detection/two_stage_detector.ipynb", "assignments/a4-object-detection"),
    ("assignments/a5-rnn-transformers/rnn_lstm_captioning.ipynb", "assignments/a5-rnn-transformers"),
    ("assignments/a5-rnn-transformers/Transformers.ipynb", "assignments/a5-rnn-transformers"),
]


def source(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def colab_or_local_setup(assignment_dir: str) -> tuple[list[str], list[str]]:
    clone_cell = f"""
# Google Drive mounting is not needed. In Colab this notebook clones the public
# GitHub repository; in local JupyterLab it uses the current assignment folder.
import os
import subprocess

IN_COLAB = os.path.isdir("/content")
if IN_COLAB:
    if not os.path.isdir("{COLAB_REPO}"):
        subprocess.run(
            ["git", "clone", "--depth", "1", "{REPO_URL}", "{COLAB_REPO}"],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", "{COLAB_REPO}", "pull", "--ff-only"], check=True)
else:
    print("Running locally; using the current assignment directory.")
"""

    path_cell = f"""
import os
import sys

colab_assignment_path = "{COLAB_REPO}/{assignment_dir}"
if os.path.isdir(colab_assignment_path):
    GOOGLE_DRIVE_PATH = colab_assignment_path
else:
    GOOGLE_DRIVE_PATH = os.getcwd()

GOOGLE_DRIVE_PATH = os.path.abspath(GOOGLE_DRIVE_PATH)
os.chdir(GOOGLE_DRIVE_PATH)
if GOOGLE_DRIVE_PATH not in sys.path:
    sys.path.insert(0, GOOGLE_DRIVE_PATH)

print(os.listdir(GOOGLE_DRIVE_PATH))
"""
    return source(clone_cell), source(path_cell)


def replace_coco_wget(src: str) -> str:
    old = "    !wget --no-check-certificate https://web.eecs.umich.edu/~justincj/teaching/eecs498/coco.pt -P ./datasets/"
    if old not in src:
        return src

    new = """    import shutil
    import ssl
    import urllib.request

    os.makedirs("./datasets", exist_ok=True)
    url = "https://web.eecs.umich.edu/~justincj/teaching/eecs498/coco.pt"
    output_path = "./datasets/coco.pt"
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=context) as response:
        with open(output_path, "wb") as output_file:
            shutil.copyfileobj(response, output_file)"""
    return src.replace(old, new)


def prepare_notebook(path: Path, assignment_dir: str) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
    if len(code_cells) < 3:
        raise ValueError(f"Expected at least three code cells in {path}")

    cell1, cell2 = colab_or_local_setup(assignment_dir)
    changed = False

    if code_cells[1].get("source") != cell1:
        code_cells[1]["source"] = cell1
        changed = True
    if code_cells[2].get("source") != cell2:
        code_cells[2]["source"] = cell2
        changed = True

    for cell in code_cells:
        src = "".join(cell.get("source", []))
        updated = src.replace('time.tzset()', 'getattr(time, "tzset", lambda: None)()')
        updated = replace_coco_wget(updated)
        if updated != src:
            cell["source"] = source(updated)
            changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    for rel_path, assignment_dir in NOTEBOOKS:
        path = REPO_ROOT / rel_path
        changed = prepare_notebook(path, assignment_dir)
        print(f"{rel_path}: {'updated' if changed else 'unchanged'}")


if __name__ == "__main__":
    main()
