"""Execute the authors' toy notebook without mutating the original file."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


INSTALL_PREFIXES = ("%pip install", "!pip install", "pip install")
LOCAL_VALIDATION_SOURCE = """# LOCAL_VALIDATION: finite one-pass samples from both trained models
for _name, _model in (("swiss_roll", model_swiss), ("checkerboard", model_checker)):
    _model.eval()
    with torch.no_grad():
        _samples = _model(torch.randn(64, 32, device=DEVICE))
    assert tuple(_samples.shape) == (64, 2), (_name, tuple(_samples.shape))
    assert bool(torch.isfinite(_samples).all()), f"{_name} produced non-finite samples"
    print(f"LOCAL_VALIDATION {_name}: shape={tuple(_samples.shape)} finite=True")
"""


def prepare_notebook(
    source: Path,
    *,
    include_validation: bool = False,
) -> nbformat.NotebookNode:
    """Return an execution copy with Colab dependency-install cells disabled."""
    notebook = nbformat.read(source, as_version=4)
    prepared = copy.deepcopy(notebook)

    for index, cell in enumerate(prepared.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        first_line = cell.source.lstrip().splitlines()[0].strip()
        if first_line.startswith(INSTALL_PREFIXES):
            prepared.cells[index] = nbformat.v4.new_markdown_cell(
                "Local dependency installation skipped; use `requirements-toy.txt`."
            )

    if include_validation:
        prepared.cells.append(nbformat.v4.new_code_cell(LOCAL_VALIDATION_SOURCE))

    return prepared


def execute_notebook(
    source: Path,
    output: Path,
    *,
    kernel_name: str = "python3",
    timeout: int = 3600,
) -> Path:
    """Execute a prepared copy of ``source`` and write it to ``output``."""
    prepared = prepare_notebook(source, include_validation=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        prepared,
        kernel_name=kernel_name,
        timeout=timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(source.resolve().parent)}},
    )
    executed = client.execute()
    nbformat.write(executed, output)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("notebooks/drifting_model_demo_original.ipynb"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/toy/drifting_model_demo_executed.ipynb"),
    )
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = execute_notebook(
        args.source,
        args.output,
        kernel_name=args.kernel_name,
        timeout=args.timeout,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
