"""Tests for the multimodal analysis endpoints (Interactive Analyst)."""

import io
import wave

import numpy as np


def _png_bytes() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (120, 80), (200, 90, 60))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _wav_bytes(freq: float = 440.0, seconds: float = 0.5) -> bytes:
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    pcm = (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_analyze_image(test_client):
    resp = test_client.post(
        "/api/v1/media/analyze",
        files={"file": ("photo.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "image"
    assert body["measurements"]["width"] == 120
    assert body["measurements"]["height"] == 80
    assert "🖼️" in body["markdown"]
    assert "|" in body["markdown"]  # tables present
    assert len(body["highlights"]) >= 2


def test_capture_photo_endpoint(test_client):
    resp = test_client.post(
        "/api/v1/media/capture/photo",
        files={"file": ("live-photo.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["kind"] == "image"


def test_capture_audio_measures_real_frequency(test_client):
    resp = test_client.post(
        "/api/v1/media/capture/audio",
        files={"file": ("live-audio.wav", _wav_bytes(440.0), "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "audio"
    # The dominant frequency must be measured from the actual samples.
    assert abs(body["measurements"]["dominant_frequency_hz"] - 440.0) < 5.0
    assert "🎙️" in body["markdown"]


def test_analyze_csv_dataset(test_client):
    csv = b"x,y\n1,2\n2,4\n3,6\n4,8\n"
    resp = test_client.post(
        "/api/v1/media/analyze",
        files={"file": ("data.csv", csv, "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "tabular"
    assert body["measurements"]["rows"] == 4
    # Perfect correlation must be detected from the real cells.
    corr = body["measurements"]["top_correlations"]
    assert corr and abs(float(corr[0]["r"])) == 1.0


def test_analyze_text_document(test_client):
    text = b"# Notes\nAlpha beta gamma. Alpha delta epsilon. Alpha zeta."
    resp = test_client.post(
        "/api/v1/media/analyze",
        files={"file": ("notes.md", text, "text/markdown")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "text"
    top = body["measurements"]["top_keywords"][0]
    assert top["word"] == "alpha"
    assert top["count"] == 3


def test_analyze_and_index_adds_chunks(test_client):
    csv = b"a,b\n1,2\n3,4\n"
    resp = test_client.post(
        "/api/v1/media/analyze/index",
        files={"file": ("data.csv", csv, "text/csv")},
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] >= 1


def test_unsupported_format_rejected(test_client):
    resp = test_client.post(
        "/api/v1/media/analyze",
        files={"file": ("archive.zip", b"PK\x03\x04junk", "application/zip")},
    )
    assert resp.status_code == 400


def test_corrupt_wav_rejected(test_client):
    resp = test_client.post(
        "/api/v1/media/capture/audio",
        files={"file": ("bad.wav", b"not-a-wav", "audio/wav")},
    )
    assert resp.status_code == 422


def test_empty_file_rejected(test_client):
    resp = test_client.post(
        "/api/v1/media/analyze",
        files={"file": ("photo.png", b"", "image/png")},
    )
    assert resp.status_code == 400
