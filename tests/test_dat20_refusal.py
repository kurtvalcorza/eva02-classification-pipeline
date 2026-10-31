"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

from eva02_classification_pipeline import pipeline as pipeline_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/eva02_classification_colab.ipynb"
BYOD_CELL = "eva02_classification-13"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = "eva02_classification"
DATA_CELL = f"{STEM}-21"
IMAGE_EARLIER = {"image": "old", "result": "old", "report": "old", "input_manifest": "old", "ground_truth": "old"}
DATA_EARLIER = {"dataset": "old", "train_images": "old", "fine_tuned_pipe": "old", "finetuned_eval_report": "old", "run_history": "old", "CUSTOM_CLASSES": "old"}


def _png(color=(120, 30, 30), size=(32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_finetuned").mkdir(parents=True)
    for name in ("input_manifest.json", "evaluation_report.json", "result.json", "top_k.csv", "finetuned_evaluation_report.json", "validation_predictions.csv"):
        (out / f"{STEM}_{name}").write_text("x", encoding="utf-8")


def _colab(uploaded: dict):
    module = types.ModuleType("google.colab")
    module.files = types.SimpleNamespace(upload=lambda: uploaded)
    return module


def _image(monkeypatch, tmp_path, path: str | None, *, use: bool = True, colab=None, source: str | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _seed_outputs(tmp_path)
    source = source or _cell(BYOD_CELL)
    if use:
        source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1)
    if path is not None:
        source = source.replace("BYOD_IMAGE_PATH = ''", f"BYOD_IMAGE_PATH = {path!r}", 1)
    ns = {"NUM_CLASSES": 1000, **IMAGE_EARLIER}
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _dataset(monkeypatch, tmp_path, path: str | None, *, colab=None, full: bool = False) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _seed_outputs(tmp_path)
    source = _cell(DATA_CELL).replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True", 1)
    if not full:
        source = source[: source.index("CUSTOM_CLASSES = dataset['classes']")]
    if path is not None:
        source = source.replace("BYOD_DATASET_PATH = ''", f"BYOD_DATASET_PATH = {path!r}", 1)
    ns = {"load_image_zip": pipeline_module.load_image_zip, **DATA_EARLIER}
    exec(compile(source, "<section 8>", "exec"), ns)
    return ns


def _assert_refused(caught, file_name, rule, tmp_path, kept):
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == sorted(f"{STEM}_{k}" for k in kept)


IMAGE_KEPT = ["finetuned", "finetuned_evaluation_report.json", "validation_predictions.csv"]
DATA_KEPT = ["input_manifest.json", "evaluation_report.json"]


# --- Section 4: image -----------------------------------------------------------------------------------------------


def _image_cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    return {
        "text file (the scripted /etc/hostname run)": ("hostname", "not a decodable image"),
        "folder": ("folder", "does not exist or is not a file"),
        "missing path": ("nope.png", "does not exist or is not a file"),
    }


@pytest.mark.parametrize("case", ["text file (the scripted /etc/hostname run)", "folder", "missing path"])
def test_image_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _image_cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _image(monkeypatch, tmp_path, str(tmp_path / file_name))
    _assert_refused(caught, file_name, rule, tmp_path, IMAGE_KEPT)


def test_image_refusal_clears_the_session(monkeypatch, tmp_path):
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    source = _cell(BYOD_CELL).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_IMAGE_PATH = ''", "BYOD_IMAGE_PATH = 'notes.txt'", 1)
    ns = {"NUM_CLASSES": 1000, "pipe": "kept", **IMAGE_EARLIER, **DATA_EARLIER}
    with pytest.raises(ValueError, match="^notes.txt: not a decodable image"):
        exec(compile(source, "<section 4>", "exec"), ns)
    assert not set(IMAGE_EARLIER) & set(ns), "an earlier image run's results survived the refusal"
    assert ns["pipe"] == "kept" and set(DATA_EARLIER) <= set(ns)


def test_image_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _image(monkeypatch, tmp_path, None)
    _assert_refused(caught, "(no file)", "there is no upload dialog outside Google Colab", tmp_path, IMAGE_KEPT)


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "got 0"), ({"a.png": _png(), "b.png": _png()}, "a.png, b.png", "got 2")])
def test_cancelled_or_multiple_image_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    with pytest.raises(ValueError) as caught:
        _image(monkeypatch, tmp_path, None, colab=_colab(uploaded))
    _assert_refused(caught, file_name, rule, tmp_path, IMAGE_KEPT)


def test_good_image_is_accepted(monkeypatch, tmp_path):
    (tmp_path / "ok.png").write_bytes(_png())
    ns = _image(monkeypatch, tmp_path, str(tmp_path / "ok.png"))
    assert ns["sample_kind"] == "BYOD" and ns["image"].size == (32, 32)


def test_image_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "ok.png").write_bytes(_png())
    source = _cell(BYOD_CELL).replace("            image.load()\n", "            raise RuntimeError('internal bug')\n", 1)
    assert "internal bug" in source  # raised inside the wrap, which must let it through
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _image(monkeypatch, tmp_path, str(tmp_path / "ok.png"), source=source)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_IMAGE_PATH='/etc/hostname': a file present in every fresh runtime."""
    with pytest.raises(ValueError) as caught:
        _image(monkeypatch, tmp_path, "/etc/hostname")
    _assert_refused(caught, "hostname", "not a decodable image (UnidentifiedImageError)", tmp_path, IMAGE_KEPT)


def test_default_image_run_is_unchanged(monkeypatch, tmp_path):
    ns = _image(monkeypatch, tmp_path, None, use=False)
    assert ns["sample_kind"] == "synthetic" and ns["result"] == "old"


# --- Section 8: dataset ---------------------------------------------------------------------------------------------


def _good_entries() -> dict:
    colors = {"red": (200, 10, 10), "blue": (10, 10, 200)}
    return {f"{cls}/{i}.png": _png((c[0] + i, c[1] + i, c[2])) for cls, c in colors.items() for i in range(6)}


def _dataset_cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    (tmp_path / "unsafe.zip").write_bytes(_zip({"../evil.png": _png()}))
    (tmp_path / "one_class.zip").write_bytes(_zip({f"red/{i}.png": _png((i, 0, 0)) for i in range(6)}))
    (tmp_path / "no_images.zip").write_bytes(_zip({"notes.txt": b"x"}))
    return {
        "not a .zip (the scripted /etc/hostname run)": ("hostname", "must be a .zip file"),
        "missing": ("nope.zip", "does not exist or is not a file"),
        "corrupt zip": ("fake.zip", "not a readable .zip archive"),
        "unsafe member": ("unsafe.zip", "illegal path"),
        "one class": ("one_class.zip", "at least 2 classes"),
        "no images": ("no_images.zip", "no image files"),
    }


DATA_CASES = ["not a .zip (the scripted /etc/hostname run)", "missing", "corrupt zip", "unsafe member", "one class", "no images"]


@pytest.mark.parametrize("case", DATA_CASES)
def test_dataset_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _dataset_cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _dataset(monkeypatch, tmp_path, str(tmp_path / file_name))
    _assert_refused(caught, file_name, rule, tmp_path, DATA_KEPT)


def test_dataset_refusal_clears_the_session(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _dataset(monkeypatch, tmp_path, str(tmp_path / "nope.zip"))
    monkeypatch.chdir(tmp_path)
    source = _cell(DATA_CELL).replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True", 1).replace("BYOD_DATASET_PATH = ''", "BYOD_DATASET_PATH = 'nope.zip'", 1)
    ns = {"load_image_zip": pipeline_module.load_image_zip, "image": "kept", **DATA_EARLIER}
    with pytest.raises(ValueError, match="^nope.zip: BYOD_DATASET_PATH"):
        exec(compile(source, "<section 8>", "exec"), ns)
    assert not set(DATA_EARLIER) & set(ns), "an earlier fine-tuning run's results survived the refusal"
    assert ns["image"] == "kept"


@pytest.mark.parametrize(("colab", "file_name", "rule"), [(None, "(no file)", "there is no upload dialog outside Google Colab"), ({}, "(no file)", "got 0")])
def test_dataset_upload_problems_are_one_refused_line(monkeypatch, tmp_path, colab, file_name, rule):
    with pytest.raises(ValueError) as caught:
        _dataset(monkeypatch, tmp_path, None, colab=None if colab is None else _colab(colab))
    _assert_refused(caught, file_name, rule, tmp_path, DATA_KEPT)


def test_valid_archive_still_loads_and_runs_the_whole_cell(monkeypatch, tmp_path):
    (tmp_path / "ok.zip").write_bytes(_zip(_good_entries()))
    ns = _dataset(monkeypatch, tmp_path, str(tmp_path / "ok.zip"), full=True)
    assert ns["dataset_kind"] == "BYOD" and ns["CUSTOM_CLASSES"] == ["blue", "red"]


def test_dataset_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "ok.zip").write_bytes(_zip(_good_entries()))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    _seed_outputs(tmp_path)
    source = _cell(DATA_CELL).replace("USE_BYOD_DATASET = False", "USE_BYOD_DATASET = True", 1).replace("BYOD_DATASET_PATH = ''", "BYOD_DATASET_PATH = 'ok.zip'", 1)

    def broken(*args, **kwargs):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        exec(compile(source, "<section 8>", "exec"), {"load_image_zip": broken})
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()
