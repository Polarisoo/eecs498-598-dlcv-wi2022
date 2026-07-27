from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class LoggingNotebookClient(NotebookClient):
    async def async_execute_cell(
        self, cell, cell_index, execution_count=None, store_history=True
    ):
        if cell.get("cell_type") == "code":
            print(f"--- Running code cell {cell_index} ---", flush=True)
        result = await super().async_execute_cell(
            cell, cell_index, execution_count, store_history
        )
        if cell.get("cell_type") == "code":
            print(f"--- Finished code cell {cell_index} ---", flush=True)
            nbformat.write(self.nb, self.checkpoint_path)
        return result


def execute_notebook(
    rel_path: str,
    kernel_name: str,
    timeout: int,
    cell_indices: list[int] | None = None,
) -> None:
    path = REPO_ROOT / rel_path
    print(f"\n=== Executing {rel_path} ===", flush=True)
    notebook = nbformat.read(path, as_version=4)
    notebook.setdefault("metadata", {}).setdefault("kernelspec", {})
    notebook["metadata"]["kernelspec"].update(
        {
            "display_name": "Python (eecs498-d2l CUDA)",
            "language": "python",
            "name": kernel_name,
        }
    )
    client = LoggingNotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    client.checkpoint_path = path
    if cell_indices is None:
        client.execute()
    else:
        print(f"=== Selected cells: {cell_indices} ===", flush=True)
        with client.setup_kernel():
            for cell_index in cell_indices:
                client.execute_cell(notebook.cells[cell_index], cell_index)
    nbformat.write(notebook, path)
    print(f"=== Finished {rel_path} ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+")
    parser.add_argument("--kernel", default="eecs498-d2l")
    parser.add_argument("--timeout", type=int, default=-1)
    parser.add_argument(
        "--cells",
        help="Comma-separated zero-based cell indices to execute instead of all cells.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    cell_indices = None
    if args.cells:
        cell_indices = [int(value) for value in args.cells.split(",")]
    for notebook in args.notebooks:
        execute_notebook(notebook, args.kernel, args.timeout, cell_indices)


if __name__ == "__main__":
    main()
