"""
Interactive Analyst Agent (agent #7).

Takes REAL measurements produced by app.tools.media_analyzer (image pixels,
audio PCM, tabular cells, document text) and presents them as an
interactive, emoji-annotated Markdown briefing with tables, verdicts, and
suggested follow-ups. It never invents numbers — every figure in its output
comes from the measurement dict it receives.
"""

import time
from typing import Any, Dict, List

from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


def _bar(ratio: float, width: int = 10) -> str:
    """Unicode meter for a 0-1 ratio."""
    ratio = max(0.0, min(1.0, ratio))
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"


class InteractiveAnalystAgent:
    """Formats measured media analytics into interactive Markdown briefings."""

    name = "interactive_analyst"

    def render(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """Produce an interactive briefing for one measurement payload."""
        start = time.perf_counter()
        kind = measurements.get("kind", "unknown")
        renderers = {
            "image": self._render_image,
            "audio": self._render_audio,
            "tabular": self._render_tabular,
            "text": self._render_text,
        }
        renderer = renderers.get(kind)
        if renderer is None:
            raise ValueError(f"Unsupported measurement kind: {kind}")
        markdown, highlights, followups = renderer(measurements)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[InteractiveAnalyst] Rendered %s briefing in %.1fms", kind, elapsed_ms
        )
        return {
            "agent": self.name,
            "kind": kind,
            "markdown": markdown,
            "highlights": highlights,
            "followups": followups,
            "measurements": measurements,
            "method": (
                "all figures computed from raw input bytes by "
                "app.tools.media_analyzer; formatting only in this agent"
            ),
        }

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    def _render_image(self, m: Dict[str, Any]) -> "tuple[str, List[str], List[str]]":
        brightness = float(m["brightness"])
        contrast = float(m["contrast"])
        sharpness = float(m["sharpness"])

        exposure = (
            "🌑 Very dark"
            if brightness < 0.2
            else "🌘 Dark / moody"
            if brightness < 0.35
            else "🌗 Balanced"
            if brightness < 0.65
            else "🌕 Bright"
            if brightness < 0.85
            else "☀️ Very bright / risk of blowout"
        )
        contrast_v = (
            "😴 Flat"
            if contrast < 0.12
            else "🙂 Moderate"
            if contrast < 0.22
            else "🔥 Punchy"
        )
        sharp_v = (
            "🌫️ Soft (low edge energy)"
            if sharpness < 0.04
            else "👍 Acceptable detail"
            if sharpness < 0.10
            else "🔪 Crisp"
        )

        colors = m.get("dominant_colors", [])
        color_rows = "\n".join(
            f"| `{c['hex']}` | {_bar(float(c['share']))} | {float(c['share']):.0%} |"
            for c in colors
        )

        exif = m.get("exif", {})
        exif_line = (
            f"📷 Captured with **{exif.get('camera_make', '')} "
            f"{exif.get('camera_model', '')}**"
            + (f" on {exif['captured_at']}" if "captured_at" in exif else "")
            if exif
            else "📷 No camera metadata (EXIF) embedded — typical for live captures."
        )

        md = f"""## 🖼️ Image Analysis — `{m["filename"]}`

> {exposure} · {contrast_v} contrast · {sharp_v}

### 📐 Geometry
| Property | Value |
|---|---|
| 🖥️ Resolution | **{m["width"]} × {m["height"]}** ({m["megapixels"]} MP) |
| 📏 Aspect ratio | {m["aspect_ratio"]} |
| 🎞️ Format / mode | {m["format"]} / {m["mode"]} |
| 💾 File size | {_fmt_bytes(int(m["size_bytes"]))} |

### 💡 Tonal Profile (measured from pixels)
| Metric | Meter | Value | Verdict |
|---|---|---|---|
| Brightness | {_bar(brightness)} | {brightness:.0%} | {exposure} |
| Contrast | {_bar(min(contrast * 3, 1.0))} | {contrast:.0%} | {contrast_v} |
| Sharpness | {_bar(min(sharpness * 8, 1.0))} | {sharpness:.1%} | {sharp_v} |
| Colorfulness | {_bar(min(float(m["colorfulness"]) / 120.0, 1.0))} | {m["colorfulness"]} | {"🎨 Vivid" if float(m["colorfulness"]) > 60 else "🖤 Muted"} |

### 🎨 Dominant Palette
| Color | Share | % |
|---|---|---|
{color_rows if color_rows else "| — | — | — |"}

{exif_line}
"""
        highlights = [
            f"🖥️ {m['width']}×{m['height']} ({m['megapixels']} MP)",
            f"💡 Brightness {brightness:.0%} — {exposure.split(' ', 1)[1]}",
            f"🎨 {len(colors)} dominant colors, lead {colors[0]['hex'] if colors else 'n/a'}",
        ]
        followups = [
            "Compare this frame against a second capture",
            "Index this image's measurements into the knowledge base",
            "Explain how the sharpness metric is computed",
        ]
        return md, highlights, followups

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _render_audio(self, m: Dict[str, Any]) -> "tuple[str, List[str], List[str]]":
        rms = float(m["rms_dbfs"])
        activity = float(m["activity_ratio"])
        loud_v = (
            "🤫 Very quiet"
            if rms < -40
            else "🙂 Comfortable level"
            if rms < -18
            else "📢 Hot signal"
            if rms < -8
            else "🚨 Near clipping"
        )
        act_v = (
            "🌵 Mostly silence"
            if activity < 0.2
            else "💬 Intermittent speech/sound"
            if activity < 0.6
            else "🗣️ Continuously active"
        )
        clip_v = (
            "⚠️ **Clipping detected** — reduce input gain"
            if m["clipping"]
            else "✅ No clipping"
        )

        md = f"""## 🎙️ Audio Analysis — `{m["filename"]}`

> {loud_v} · {act_v} · {clip_v}

### 📊 Signal Measurements
| Metric | Meter | Value |
|---|---|---|
| ⏱️ Duration | — | **{m["duration_seconds"]} s** |
| 🎚️ Loudness (RMS) | {_bar((rms + 60) / 60)} | {rms} dBFS |
| ⛰️ Peak level | {_bar((float(m["peak_dbfs"]) + 60) / 60)} | {m["peak_dbfs"]} dBFS |
| 🗣️ Activity ratio | {_bar(activity)} | {activity:.0%} of windows |
| ✂️ Sound segments | — | {m["speech_segments"]} |
| 🎼 Dominant frequency | — | {m["dominant_frequency_hz"]} Hz |

### 🔧 Recording Format
| Property | Value |
|---|---|
| 📻 Sample rate | {m["sample_rate_hz"]} Hz |
| 🔢 Bit depth | {m["bit_depth"]}-bit |
| 🎧 Channels | {m["channels"]} |
| 💾 Size | {_fmt_bytes(int(m["size_bytes"]))} |

**Verdict:** {loud_v}. {act_v}. {clip_v}.
"""
        highlights = [
            f"⏱️ {m['duration_seconds']}s recording at {m['sample_rate_hz']} Hz",
            f"🎚️ RMS {rms} dBFS — {loud_v.split(' ', 1)[1]}",
            f"🗣️ {activity:.0%} active, {m['speech_segments']} segment(s)",
        ]
        followups = [
            "Record a longer sample for a steadier loudness estimate",
            "Explain what dBFS and activity ratio mean",
            "Check whether this level suits voice notes or podcasting",
        ]
        return md, highlights, followups

    # ------------------------------------------------------------------
    # Tabular
    # ------------------------------------------------------------------

    def _render_tabular(self, m: Dict[str, Any]) -> "tuple[str, List[str], List[str]]":
        missing = float(m["missing_ratio"])
        quality = (
            "💎 Complete — no missing cells"
            if missing == 0
            else "✅ Healthy (<2% missing)"
            if missing < 0.02
            else "🩹 Patchy (2–10% missing)"
            if missing < 0.10
            else "🚧 Sparse — significant gaps"
        )

        num_rows = "\n".join(
            f"| `{s['column']}` | {s['mean']} | {s['min']} | {s['max']} | "
            f"{'✅ 0' if s['missing'] == 0 else '⚠️ ' + str(s['missing'])} |"
            for s in m.get("numeric_summary", [])
        )
        cat_rows = "\n".join(
            f"| `{s['column']}` | {s['unique']} | "
            f"{', '.join(v[:18] for v in s['top_values'])} |"
            for s in m.get("categorical_summary", [])
        )
        corr_rows = "\n".join(
            f"| {c['pair']} | {_bar(abs(float(c['r'])))} | **{c['r']}** | "
            f"{'📈 strong' if abs(float(c['r'])) > 0.7 else '↔️ moderate' if abs(float(c['r'])) > 0.4 else '🌫️ weak'} |"
            for c in m.get("top_correlations", [])
        )

        preview = m.get("preview", {})
        p_cols = preview.get("columns", [])
        p_rows = preview.get("rows", [])
        preview_md = ""
        if p_cols:
            preview_md = (
                "| "
                + " | ".join(p_cols)
                + " |\n"
                + "|"
                + "---|" * len(p_cols)
                + "\n"
                + "\n".join("| " + " | ".join(r) + " |" for r in p_rows)
            )

        md = f"""## 📊 Dataset Analysis — `{m["filename"]}`

> 🧮 **{m["rows"]:,} rows × {m["columns"]} columns** · {quality}

### 🧾 Schema at a Glance
| Property | Value |
|---|---|
| 🔢 Numeric columns | {m["numeric_columns"]} |
| 🔤 Categorical columns | {m["categorical_columns"]} |
| 🕳️ Missing cells | {m["missing_cells"]:,} ({missing:.1%}) |
| 💾 Size | {_fmt_bytes(int(m["size_bytes"]))} |

{"### 🔢 Numeric Columns" if num_rows else ""}
{"| Column | Mean | Min | Max | Missing |" if num_rows else ""}
{"|---|---|---|---|---|" if num_rows else ""}
{num_rows}

{"### 🔤 Categorical Columns" if cat_rows else ""}
{"| Column | Unique | Top values |" if cat_rows else ""}
{"|---|---|---|" if cat_rows else ""}
{cat_rows}

{"### 🔗 Strongest Correlations (Pearson r)" if corr_rows else ""}
{"| Pair | Strength | r | Reading |" if corr_rows else ""}
{"|---|---|---|---|" if corr_rows else ""}
{corr_rows}

{"### 👀 First 5 Rows" if preview_md else ""}
{preview_md}
"""
        top_corr = m.get("top_correlations", [])
        highlights = [
            f"🧮 {m['rows']:,} rows × {m['columns']} cols",
            f"🕳️ {missing:.1%} missing — {quality.split(' ', 1)[1] if ' ' in quality else quality}",
        ]
        if top_corr:
            highlights.append(
                f"🔗 Strongest link: {top_corr[0]['pair']} (r={top_corr[0]['r']})"
            )
        followups = [
            "Index this dataset into the knowledge base for Q&A",
            "Explain the strongest correlation found",
            "Suggest cleaning steps for the missing cells",
        ]
        return md, highlights, followups

    # ------------------------------------------------------------------
    # Text / documents
    # ------------------------------------------------------------------

    def _render_text(self, m: Dict[str, Any]) -> "tuple[str, List[str], List[str]]":
        keywords = m.get("top_keywords", [])
        max_count = max((int(k["count"]) for k in keywords), default=1)
        kw_rows = "\n".join(
            f"| {i + 1} | **{k['word']}** | {_bar(int(k['count']) / max_count)} | {k['count']} |"
            for i, k in enumerate(keywords)
        )
        density = float(m.get("unique_word_ratio", 0.0))
        density_v = (
            "🔁 Repetitive vocabulary"
            if density < 0.3
            else "🙂 Typical variety"
            if density < 0.6
            else "🌈 Rich vocabulary"
        )

        md = f"""## 📄 Document Analysis — `{m["filename"]}`

> 📚 **{m["words"]:,} words** · ⏳ ~{m["reading_time_minutes"]} min read · {density_v}

### 🧾 Structure
| Metric | Value |
|---|---|
| 🔤 Characters | {m["characters"]:,} |
| 📚 Words | {m["words"]:,} |
| ✂️ Sentences | {m["sentences"]:,} |
| 🏷️ Headings | {m["headings"]} |
| 📏 Avg sentence length | {m["avg_sentence_length"]} words |
| 🌈 Unique-word ratio | {density:.0%} — {density_v} |

### 🔑 Top Keywords
| # | Keyword | Frequency | Count |
|---|---|---|---|
{kw_rows if kw_rows else "| — | — | — | — |"}
"""
        highlights = [
            f"📚 {m['words']:,} words (~{m['reading_time_minutes']} min read)",
            f"🔑 Top keyword: {keywords[0]['word'] if keywords else 'n/a'}",
            f"🌈 Vocabulary: {density:.0%} unique — {density_v}",
        ]
        followups = [
            "Index this document into the knowledge base",
            "Run a research query grounded in this document",
            "Summarize the document's key sections",
        ]
        return md, highlights, followups
