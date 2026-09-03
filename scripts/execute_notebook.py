#!/usr/bin/env python3
"""Execute plain Python notebook cells without requiring a Jupyter runtime."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import traceback


def execute_notebook(source: Path, destination: Path) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook__"}
    execution_count = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        output = io.StringIO()
        error = None
        source_code = cell.get("source", "")
        if isinstance(source_code, list):
            source_code = "".join(source_code)
        try:
            with redirect_stdout(output), redirect_stderr(output):
                exec(compile(source_code, f"{source}#{cell.get('id', execution_count)}", "exec"), namespace)
        except BaseException as exc:
            error = exc
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                }
            ]
        else:
            text = output.getvalue()
            cell["outputs"] = (
                [{"output_type": "stream", "name": "stdout", "text": text}]
                if text
                else []
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        if error is not None:
            raise error


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    execute_notebook(args.source, args.destination)


if __name__ == "__main__":
    main()
