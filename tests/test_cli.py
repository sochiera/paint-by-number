import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from pbn.cli import main


def _write_stripe_image(path: Path):
    img = np.zeros((24, 30, 3), dtype=np.uint8)
    img[:, :10] = (255, 0, 0)
    img[:, 10:20] = (0, 255, 0)
    img[:, 20:] = (0, 0, 255)
    Image.fromarray(img).save(path)


def test_cli_main_writes_expected_files(tmp_path):
    inp = tmp_path / "in.png"
    _write_stripe_image(inp)
    outdir = tmp_path / "out"

    exit_code = main(
        [str(inp), "-o", str(outdir), "-k", "3", "--min-region", "2", "--scale", "2"]
    )
    assert exit_code == 0

    assert (outdir / "preview.png").exists()
    assert (outdir / "template.png").exists()
    assert (outdir / "legend.png").exists()
    assert (outdir / "palette.json").exists()

    palette_data = json.loads((outdir / "palette.json").read_text())
    assert palette_data["k"] == 3
    assert len(palette_data["colors"]) == 3
    for entry in palette_data["colors"]:
        assert "index" in entry and "rgb" in entry
        assert len(entry["rgb"]) == 3


def test_cli_rejects_bad_k(tmp_path):
    import pytest

    inp = tmp_path / "in.png"
    _write_stripe_image(inp)
    outdir = tmp_path / "out"
    # argparse's error() raises SystemExit with a non-zero code.
    with pytest.raises(SystemExit) as excinfo:
        main([str(inp), "-o", str(outdir), "-k", "0"])
    assert excinfo.value.code != 0


def test_smooth_flag_accepts_bilateral(tmp_path):
    inp = tmp_path / "in.png"
    _write_stripe_image(inp)
    outdir = tmp_path / "out_bilateral"

    exit_code = main(
        [
            str(inp),
            "-o",
            str(outdir),
            "-k",
            "3",
            "--min-region",
            "2",
            "--scale",
            "2",
            "--smooth",
            "bilateral",
        ]
    )
    assert exit_code == 0
    assert (outdir / "preview.png").exists()


def test_smooth_flag_rejects_unknown(tmp_path):
    import pytest

    inp = tmp_path / "in.png"
    _write_stripe_image(inp)
    outdir = tmp_path / "out_bogus"
    with pytest.raises(SystemExit) as excinfo:
        main([str(inp), "-o", str(outdir), "--smooth", "bogus"])
    assert excinfo.value.code != 0


def test_max_regions_flag(tmp_path):
    inp = tmp_path / "in.png"
    _write_stripe_image(inp)
    outdir = tmp_path / "out_maxregions"

    exit_code = main(
        [
            str(inp),
            "-o",
            str(outdir),
            "-k",
            "3",
            "--min-region",
            "2",
            "--scale",
            "2",
            "--max-regions",
            "40",
        ]
    )
    assert exit_code == 0
    assert (outdir / "preview.png").exists()
    assert (outdir / "template.png").exists()
    assert (outdir / "legend.png").exists()
    assert (outdir / "palette.json").exists()


def test_cleanup_flag(tmp_path):
    inp = tmp_path / "in.png"
    _write_stripe_image(inp)
    outdir = tmp_path / "out_cleanup_none"

    exit_code = main(
        [
            str(inp),
            "-o",
            str(outdir),
            "-k",
            "3",
            "--min-region",
            "2",
            "--scale",
            "2",
            "--cleanup",
            "none",
        ]
    )
    assert exit_code == 0
    assert (outdir / "preview.png").exists()


def test_cleanup_flag_rejects_unknown(tmp_path):
    import pytest

    inp = tmp_path / "in.png"
    _write_stripe_image(inp)
    outdir = tmp_path / "out_cleanup_bogus"
    with pytest.raises(SystemExit) as excinfo:
        main([str(inp), "-o", str(outdir), "--cleanup", "bogus"])
    assert excinfo.value.code != 0


def test_cli_as_subprocess(tmp_path):
    """Smoke-test ``python -m pbn`` so the installed entry point works."""
    inp = tmp_path / "in.png"
    _write_stripe_image(inp)
    outdir = tmp_path / "out_sub"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pbn",
            str(inp),
            "-o",
            str(outdir),
            "-k",
            "3",
            "--scale",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env={"PYTHONPATH": "src", "PATH": ""},
    )
    assert result.returncode == 0, result.stderr
    assert (outdir / "preview.png").exists()
