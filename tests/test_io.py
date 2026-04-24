import numpy as np
import pytest
from PIL import Image

from pbn.io import load_image, save_image


def test_load_image_returns_uint8_rgb_array(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (4, 3), (200, 100, 50)).save(path)

    arr = load_image(path)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (3, 4, 3)  # H, W, 3
    assert tuple(arr[0, 0]) == (200, 100, 50)


def test_load_image_converts_rgba_to_rgb(tmp_path):
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (2, 2), (10, 20, 30, 128)).save(path)

    arr = load_image(path)

    assert arr.shape == (2, 2, 3)
    assert arr.dtype == np.uint8


def test_load_image_converts_grayscale_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (2, 2), 128).save(path)

    arr = load_image(path)

    assert arr.shape == (2, 2, 3)
    assert tuple(arr[0, 0]) == (128, 128, 128)


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_save_image_roundtrip(tmp_path):
    arr = np.zeros((5, 7, 3), dtype=np.uint8)
    arr[..., 0] = 255
    path = tmp_path / "out.png"

    save_image(arr, path)

    reloaded = load_image(path)
    assert reloaded.shape == arr.shape
    assert np.array_equal(reloaded, arr)


def test_save_image_rejects_wrong_dtype(tmp_path):
    arr = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        save_image(arr, tmp_path / "bad.png")


def test_save_image_rejects_wrong_shape(tmp_path):
    arr = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        save_image(arr, tmp_path / "bad.png")
