"""Utility functions for Med-Gemma Impact Challenge."""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_sample_xray(size: int = 512) -> Image.Image:
    """Create a sample chest X-ray like image."""
    img = Image.new("RGB", (size, size), "black")
    pixels = img.load()

    for i in range(size):
        for j in range(size):
            if 200 < i < 312 and 150 < j < 362:
                val = 180 + int(30 * np.sin(i / 50) * np.cos(j / 50))
                pixels[i, j] = (val, val, val)
            elif 100 < i < 412 and 250 < j < 300:
                val = 100 + int(20 * np.sin(i / 30))
                pixels[i, j] = (val, val, val)
            else:
                val = 40 + int(10 * np.random.random())
                pixels[i, j] = (val, val, val)

    return img


def create_sample_ct(size: int = 512) -> Image.Image:
    """Create a sample CT scan like image."""
    img = Image.new("RGB", (size, size), "black")
    pixels = img.load()
    cx, cy = size // 2, size // 2

    for i in range(size):
        for j in range(size):
            dist = ((i - cx) ** 2 + (j - cy) ** 2) ** 0.5
            if dist < 200:
                if dist < 50:
                    val = 200
                elif dist < 150:
                    val = 150 + int(20 * np.sin(i / 20) * np.cos(j / 20))
                else:
                    val = 100 + int(30 * np.random.random())
                pixels[i, j] = (val, val, val)

    return img


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."