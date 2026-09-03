from pathlib import Path

import nbformat

from scripts.run_toy_notebook import prepare_notebook


def _make_notebook(*sources: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source) for source in sources]
    )


def test_prepare_notebook_skips_only_colab_install_cell(tmp_path: Path) -> None:
    source = tmp_path / "source.ipynb"
    original = _make_notebook("%pip install torch", "answer = 42")
    nbformat.write(original, source)

    prepared = prepare_notebook(source)

    source_after = nbformat.read(source, as_version=4)
    assert source_after.cells[0].source == "%pip install torch"
    assert prepared.cells[0].cell_type == "markdown"
    assert "requirements-toy.txt" in prepared.cells[0].source
    assert prepared.cells[1].cell_type == "code"
    assert prepared.cells[1].source == "answer = 42"


def test_prepare_notebook_preserves_non_install_shell_cells(tmp_path: Path) -> None:
    source = tmp_path / "source.ipynb"
    nbformat.write(_make_notebook("!pwd"), source)

    prepared = prepare_notebook(source)

    assert prepared.cells[0].cell_type == "code"
    assert prepared.cells[0].source == "!pwd"


def test_prepare_notebook_can_append_local_sample_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.ipynb"
    nbformat.write(_make_notebook("answer = 42"), source)

    prepared = prepare_notebook(source, include_validation=True)

    assert len(prepared.cells) == 2
    assert prepared.cells[-1].cell_type == "code"
    assert "LOCAL_VALIDATION" in prepared.cells[-1].source
    assert "torch.isfinite" in prepared.cells[-1].source
