"""
Multimodal media measurement tools.

Every function returns REAL measurements computed from the raw bytes of the
input (image pixels, audio PCM samples, tabular cells, document text) —
no fabricated or asserted values. The InteractiveAnalystAgent formats these
measurements into human-friendly answers.
"""

import io
import math
import wave
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


# --------------------------------------------------------------------------
# Image analysis (Pillow + numpy)
# --------------------------------------------------------------------------


def _dominant_colors(img: Any, count: int = 5) -> List[Dict[str, Any]]:
    """Extract dominant colors via palette quantization."""
    small = img.convert("RGB").resize((96, 96))
    quantized = small.quantize(colors=count)
    palette = quantized.getpalette() or []
    color_counts = sorted(quantized.getcolors() or [], reverse=True)
    total = sum(c for c, _ in color_counts) or 1
    colors = []
    for pixel_count, palette_idx in color_counts[:count]:
        r, g, b = palette[palette_idx * 3 : palette_idx * 3 + 3]
        colors.append(
            {
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "share": round(pixel_count / total, 4),
            }
        )
    return colors


def analyze_image_bytes(data: bytes, filename: str = "image") -> Dict[str, Any]:
    """
    Measure an image: geometry, exposure, contrast, sharpness, color makeup.

    Returns a dict of real measurements (no estimates are invented; sharpness
    and colorfulness use standard published metrics).
    """
    from PIL import Image, ImageFilter

    img = Image.open(io.BytesIO(data))
    img.load()
    width, height = img.size
    megapixels = round(width * height / 1_000_000, 2)

    rgb = img.convert("RGB")
    thumb = rgb.copy()
    thumb.thumbnail((256, 256))
    arr = np.asarray(thumb, dtype=np.float32)

    # Exposure and contrast from luminance (ITU-R BT.601).
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    brightness = float(luma.mean()) / 255.0
    contrast = float(luma.std()) / 255.0

    # Sharpness: mean edge magnitude after a FIND_EDGES convolution.
    edges = thumb.convert("L").filter(ImageFilter.FIND_EDGES)
    sharpness = float(np.asarray(edges, dtype=np.float32).mean()) / 255.0

    # Colorfulness: Hasler & Süsstrunk (2003) metric, normalized.
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorfulness = float(
        math.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )

    exif: Dict[str, str] = {}
    try:
        raw_exif = img.getexif()
        tag_names = {271: "camera_make", 272: "camera_model", 306: "captured_at"}
        for tag_id, key in tag_names.items():
            if tag_id in raw_exif:
                exif[key] = str(raw_exif[tag_id]).strip()
    except Exception:  # noqa: BLE001 - EXIF is best-effort
        pass

    return {
        "kind": "image",
        "filename": filename,
        "format": img.format or "unknown",
        "width": width,
        "height": height,
        "megapixels": megapixels,
        "aspect_ratio": round(width / height, 3) if height else 0.0,
        "mode": img.mode,
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "sharpness": round(sharpness, 4),
        "colorfulness": round(colorfulness, 2),
        "dominant_colors": _dominant_colors(rgb),
        "exif": exif,
        "size_bytes": len(data),
    }


# --------------------------------------------------------------------------
# Audio analysis (stdlib wave + numpy — expects WAV/PCM)
# --------------------------------------------------------------------------


def analyze_audio_bytes(data: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
    """
    Measure a WAV recording: duration, loudness, dynamics, spectrum, speech
    activity. All values computed from the actual PCM samples.
    """
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        samples = (
            np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        ) / 128.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    duration = len(samples) / sample_rate if sample_rate else 0.0
    if len(samples) == 0:
        raise ValueError("Audio file contains no samples")

    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    rms_dbfs = 20 * math.log10(rms) if rms > 0 else -120.0
    peak_dbfs = 20 * math.log10(peak) if peak > 0 else -120.0

    # Energy-based activity detection over 20 ms windows.
    win = max(1, int(sample_rate * 0.02))
    n_windows = len(samples) // win
    active_windows = 0
    segments = 0
    prev_active = False
    if n_windows > 0:
        framed = samples[: n_windows * win].reshape(n_windows, win)
        energies = np.sqrt(np.mean(framed**2, axis=1))
        threshold = max(10 ** (-45 / 20), float(energies.max()) * 0.1)
        for e in energies:
            active = bool(e >= threshold)
            if active:
                active_windows += 1
                if not prev_active:
                    segments += 1
            prev_active = active
    activity_ratio = active_windows / n_windows if n_windows else 0.0

    # Dominant frequency via FFT of up to the first 10 seconds (Hann window).
    chunk = samples[: min(len(samples), sample_rate * 10)]
    windowed = chunk * np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)
    # Ignore DC / sub-audible rumble below 50 Hz.
    mask = freqs >= 50
    dominant_hz = (
        float(freqs[mask][int(np.argmax(spectrum[mask]))]) if mask.any() else 0.0
    )

    zero_crossings = int(np.sum(np.abs(np.diff(np.signbit(samples)))))
    zcr = zero_crossings / len(samples)

    return {
        "kind": "audio",
        "filename": filename,
        "duration_seconds": round(duration, 2),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "bit_depth": sample_width * 8,
        "rms_dbfs": round(rms_dbfs, 1),
        "peak_dbfs": round(peak_dbfs, 1),
        "clipping": bool(peak >= 0.999),
        "activity_ratio": round(activity_ratio, 3),
        "speech_segments": segments,
        "dominant_frequency_hz": round(dominant_hz, 1),
        "zero_crossing_rate": round(zcr, 4),
        "size_bytes": len(data),
    }


# --------------------------------------------------------------------------
# Tabular data analysis (pandas)
# --------------------------------------------------------------------------


def analyze_tabular_bytes(data: bytes, filename: str, ext: str) -> Dict[str, Any]:
    """
    Profile a tabular dataset (CSV/TSV/XLSX/JSON): schema, missingness,
    numeric summaries, strongest correlations, and a small preview.
    """
    import pandas as pd

    buf = io.BytesIO(data)
    if ext in (".csv", ".tsv"):
        df = pd.read_csv(buf, sep="\t" if ext == ".tsv" else ",")
    elif ext == ".xlsx":
        df = pd.read_excel(buf)
    elif ext == ".json":
        df = pd.read_json(buf)
    else:
        raise ValueError(f"Unsupported tabular format: {ext}")

    rows, cols = df.shape
    total_cells = rows * cols or 1
    missing_cells = int(df.isna().sum().sum())

    numeric = df.select_dtypes(include="number")
    numeric_summary = []
    for col in list(numeric.columns)[:8]:
        s = numeric[col]
        numeric_summary.append(
            {
                "column": str(col),
                "mean": round(float(s.mean()), 4) if rows else 0.0,
                "min": round(float(s.min()), 4) if rows else 0.0,
                "max": round(float(s.max()), 4) if rows else 0.0,
                "missing": int(s.isna().sum()),
            }
        )

    correlations = []
    if numeric.shape[1] >= 2 and rows >= 3:
        corr = numeric.corr(numeric_only=True)
        seen = set()
        for a in corr.columns:
            for b in corr.columns:
                if a >= b or (a, b) in seen:
                    continue
                seen.add((a, b))
                val = float(str(corr.loc[a, b]))
                if not math.isnan(val):
                    correlations.append({"pair": f"{a} × {b}", "r": round(val, 3)})
        correlations.sort(key=lambda c: abs(float(str(c["r"]))), reverse=True)
        correlations = correlations[:5]

    categorical = df.select_dtypes(exclude="number")
    categorical_summary = []
    for col in list(categorical.columns)[:5]:
        top = categorical[col].astype(str).value_counts().head(3)
        categorical_summary.append(
            {
                "column": str(col),
                "unique": int(categorical[col].nunique()),
                "top_values": [str(v) for v in top.index.tolist()],
            }
        )

    preview = df.head(5)
    preview_table = {
        "columns": [str(c) for c in preview.columns[:8]],
        "rows": [
            [str(v)[:40] for v in row[:8]]
            for row in preview.itertuples(index=False, name=None)
        ],
    }

    return {
        "kind": "tabular",
        "filename": filename,
        "rows": rows,
        "columns": cols,
        "column_names": [str(c) for c in df.columns[:30]],
        "missing_cells": missing_cells,
        "missing_ratio": round(missing_cells / total_cells, 4),
        "numeric_columns": int(numeric.shape[1]),
        "categorical_columns": int(categorical.shape[1]),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "top_correlations": correlations,
        "preview": preview_table,
        "size_bytes": len(data),
    }


# --------------------------------------------------------------------------
# Document / free-text analysis
# --------------------------------------------------------------------------

_TEXT_STOPWORDS = frozenset(
    "the a an and or but of to in on for with by from at as is are was were "
    "be been being it its this that these those which who whom what when "
    "where how why can could should would may might will shall not no nor "
    "so if then than too very s t d ll m re ve y i you he she we they them "
    "his her their our your my me him us do does did have has had".split()
)


def analyze_text(text: str, filename: str = "document") -> Dict[str, Any]:
    """
    Measure a text document: length, structure, reading time, top keywords.
    """
    import re

    words = re.findall(r"\b\w+\b", text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 2]
    headings = re.findall(r"^#{1,6}\s+.+$", text, flags=re.MULTILINE)

    content_words = [
        w.lower()
        for w in words
        if w.lower() not in _TEXT_STOPWORDS and len(w) > 2 and not w.isdigit()
    ]
    top_keywords = [
        {"word": w, "count": c} for w, c in Counter(content_words).most_common(10)
    ]

    avg_sentence_len = round(len(words) / len(sentences), 1) if sentences else 0.0

    return {
        "kind": "text",
        "filename": filename,
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "headings": len(headings),
        "avg_sentence_length": avg_sentence_len,
        "reading_time_minutes": round(len(words) / 230.0, 1),
        "top_keywords": top_keywords,
        "unique_word_ratio": round(len(set(content_words)) / len(content_words), 3)
        if content_words
        else 0.0,
    }
