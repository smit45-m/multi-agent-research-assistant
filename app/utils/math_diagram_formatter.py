"""
Deterministic Formula & Architecture Diagram Formatter.

Prevents raw/mangled LaTeX strings (such as `$$\\text{MaxPool}(X){i} = ...$$,$$\\mathbf{y}=F(\\mathbf{x},{W})+\\mathbf{x},$$`)
from appearing in the UI by:
1. Splitting comma-jammed `$$...$$,$$...$$` equations into clean, readable Unicode formula cards.
2. Converting raw LaTeX macros (`\\text{}`, `\\mathbf{}`, `\\qquad`, `\\frac{}{}`, `\\sum`, `\\lfloor`, `\\rfloor`, etc.)
   both inside `$$...$$` / `$...$` blocks AND bare in prose into clean, human-readable mathematical notation
   that renders cleanly in Markdown without `marked.parse()` eating `_` subscripts.
3. Ensuring technical/architectural queries always include clear visual Architecture & Tensor-Flow Diagrams.
"""
import re
from typing import List


_SUBSCRIPT_MAP = str.maketrans("0123456789+-=()aeioruvxijkmn", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵣᵤᵥₓᵢⱼₖₘₙ")
_SUPERSCRIPT_MAP = str.maketrans("0123456789+-=()niT", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵀ")


def latex_to_readable_math(expr: str) -> str:
    """
    Converts a raw LaTeX math string into clean, readable Unicode mathematical formula notation
    so that Markdown parsers never strip `_` into `<em>` tags or dump raw `\\text`, `\\qquad`, `\\mathbf` macros.
    """
    s = expr.strip()
    s = re.sub(r"^(\$\$|\\\[|\\\ indeterminate|\$)+", "", s)
    s = re.sub(r"(\$\$|\\\]|\\\)|\$)+[,;.]?$", "", s)
    s = s.strip().rstrip(",;")

    # Replace spacing commands
    s = re.sub(r"\\(qquad|quad)\b", "   │   ", s)
    s = re.sub(r"\\(,|;|!|:)", " ", s)

    # Unwrap \text{...}, \mathrm{...}, \operatorname{...}, \mathbf{...}, \boldsymbol{...}, \mathcal{...}
    for _ in range(3):
        s = re.sub(r"\\(?:text|mathrm|operatorname|mathbf|boldsymbol|mathcal|mathit)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\hat\s*\{([^{}]+)\}", r"\1̂", s)
    s = re.sub(r"\\hat\s*([A-Za-z])", r"\1̂", s)
    s = re.sub(r"\\bar\s*\{([^{}]+)\}", r"\1̄", s)
    s = re.sub(r"\\tilde\s*\{([^{}]+)\}", r"\1̃", s)

    # Convert simple superscripts ^{2}, ^{T}, ^2, ^T FIRST so nested braces inside \frac{1}{k^{2}} flatten to {k²}
    def _sup(m: re.Match) -> str:
        inner = m.group(1).strip()
        if all(c in "0123456789+-=()niT" for c in inner):
            return inner.translate(_SUPERSCRIPT_MAP)
        return f"^({inner})"

    for _ in range(3):
        s = re.sub(r"\^\s*\{([^{}]+)\}", _sup, s)
    s = re.sub(r"\^([0-9Tni])", lambda m: m.group(1).translate(_SUPERSCRIPT_MAP), s)

    # Convert simple subscripts _{i}, _{k}, _i before \frac as well
    def _sub(m: re.Match) -> str:
        inner = m.group(1).strip()
        if len(inner) <= 3 and all(c in "0123456789+-=()aeioruvxijkmn" for c in inner):
            return inner.translate(_SUBSCRIPT_MAP)
        return f"[{inner}]"

    for _ in range(3):
        s = re.sub(r"_\s*\{([^{}]+)\}", _sub, s)

    # Replace \frac{A}{B} -> (A) / (B) or A / B
    for _ in range(3):
        def _repl_frac(m: re.Match) -> str:
            num = m.group(1).strip()
            den = m.group(2).strip()
            num_str = num if len(num) <= 3 else f"({num})"
            den_str = den if len(den) <= 3 else f"({den})"
            return f"({num_str} / {den_str}) "
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", _repl_frac, s)

    # Replace \sqrt{A} -> √(A)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", lambda m: f"√({m.group(1)})" if len(m.group(1)) > 2 else f"√{m.group(1)}", s)

    # Floor / Ceiling
    s = s.replace(r"\lfloor", "⌊").replace(r"\rfloor", "⌋")
    s = s.replace(r"\lceil", "⌈").replace(r"\rceil", "⌉")
    s = s.replace(r"\left", "").replace(r"\right", "")

    # Common math symbols & Greek letters
    replacements = [
        (r"\mathbb{R}", "ℝ"),
        (r"\mathbb{N}", "ℕ"),
        (r"\mathbb{Z}", "ℤ"),
        (r"\mathbb{E}", "𝔼"),
        (r"\sum", "∑"),
        (r"\prod", "∏"),
        (r"\max", "max"),
        (r"\min", "min"),
        (r"\softmax", "softmax"),
        (r"\in", " ∈ "),
        (r"\notin", " ∉ "),
        (r"\subset", " ⊂ "),
        (r"\subseteq", " ⊆ "),
        (r"\times", " × "),
        (r"\cdot", " · "),
        (r"\circ", " ∘ "),
        (r"\oplus", " ⊕ "),
        (r"\otimes", " ⊗ "),
        (r"\odot", " ⊙ "),
        (r"\top", "ᵀ"),
        (r"\|", "‖"),
        (r"\vert", "∣"),
        (r"\mid", " ∣ "),
        (r"\rightarrow", " → "),
        (r"\to", " → "),
        (r"\leftarrow", " ← "),
        (r"\Rightarrow", " ⇒ "),
        (r"\approx", " ≈ "),
        (r"\neq", " ≠ "),
        (r"\leq", " ≤ "),
        (r"\le", " ≤ "),
        (r"\geq", " ≥ "),
        (r"\ge", " ≥ "),
        (r"\pm", " ± "),
        (r"\infty", "∞"),
        (r"\partial", "∂"),
        (r"\nabla", "∇"),
        (r"\dots", "…"),
        (r"\ldots", "…"),
        (r"\cdots", "⋯"),
        (r"\alpha", "α"),
        (r"\beta", "β"),
        (r"\Gamma", "Γ"),
        (r"\gamma", "γ"),
        (r"\Delta", "Δ"),
        (r"\delta", "δ"),
        (r"\epsilon", "ε"),
        (r"\varepsilon", "ε"),
        (r"\zeta", "ζ"),
        (r"\eta", "η"),
        (r"\Theta", "Θ"),
        (r"\theta", "θ"),
        (r"\Lambda", "Λ"),
        (r"\lambda", "λ"),
        (r"\mu", "μ"),
        (r"\pi", "π"),
        (r"\rho", "ρ"),
        (r"\Sigma", "Σ"),
        (r"\sigma", "σ"),
        (r"\tau", "τ"),
        (r"\Phi", "Φ"),
        (r"\varphi", "φ"),
        (r"\phi", "φ"),
        (r"\Psi", "Ψ"),
        (r"\psi", "ψ"),
        (r"\Omega", "Ω"),
        (r"\omega", "ω"),
        (r"\ell", "ℓ"),
        (r"\sim", " ~ "),
        (r"\{", "{"),
        (r"\}", "}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)

    # Convert simple superscripts ^{2}, ^{T}, ^2, ^T (loop for nested braces)
    def _sup(m: re.Match) -> str:
        inner = m.group(1).strip()
        if all(c in "0123456789+-=()niT" for c in inner):
            return inner.translate(_SUPERSCRIPT_MAP)
        return f"^({inner})"

    for _ in range(3):
        s = re.sub(r"\^\s*\{([^{}]+)\}", _sup, s)
    s = re.sub(r"\^([0-9Tni])", lambda m: m.group(1).translate(_SUPERSCRIPT_MAP), s)

    # Convert simple subscripts _{i}, _{k}, _{(u,v) in W_i}, _i (loop for nested braces)
    def _sub(m: re.Match) -> str:
        inner = m.group(1).strip()
        if len(inner) <= 3 and all(c in "0123456789+-=()aeioruvxijkmn" for c in inner):
            return inner.translate(_SUBSCRIPT_MAP)
        return f"[{inner}]"

    for _ in range(3):
        s = re.sub(r"_\s*\{([^{}]+)\}", _sub, s)
    s = re.sub(r"_([0-9aeioruvxijkmn])\b", lambda m: m.group(1).translate(_SUBSCRIPT_MAP), s)
    s = re.sub(r"_([A-Za-z0-9]+)", lambda m: f"[{m.group(1)}]", s)

    # Also handle cases where underscore was omitted before braces, e.g. (X){i}, max{(u,v) ∈ Wᵢ}, ∑{(u,v) ∈ Wᵢ}
    s = re.sub(r"(?<=\))\{([aeioruvxijkmn0-9]{1,3})\}", lambda m: m.group(1).translate(_SUBSCRIPT_MAP), s)
    s = re.sub(r"(max|min|∑|∏)\s*\{([^{}]+)\}", r"\1[\2] ", s)

    # Clean up remaining braces around single tokens (except set notation like {W})
    s = re.sub(r"(?<![,\s])\{([A-Za-z0-9_]+)\}", r"\1", s)
    # Normalize spacing around = and + in compact equations
    s = re.sub(r"(?<=[A-Za-z0-9\)\}ᵢⱼₖₘₙ²³ᵀ])=(?=[A-Za-z0-9\(\{])", " = ", s)
    s = re.sub(r"(?<=[A-Za-z0-9\)\}ᵢⱼₖₘₙ²³ᵀ])\+(?=[A-Za-z0-9\(\{])", " + ", s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip().rstrip(",;.")
    return s


def build_architecture_diagrams(query: str) -> str:
    """
    Builds crisp visual Architecture & Tensor-Flow Diagrams tailored to the query concepts.
    """
    q_low = query.lower()
    blocks: List[str] = []

    has_transformer = any(w in q_low for w in ("transformer", "attention", "multi-head", "self-attention", "encoder", "decoder"))
    has_cnn_ops = any(w in q_low for w in ("stride", "padding", "pading", "pooling", "conv", "cnn", "kernel"))
    has_resnet = any(w in q_low for w in ("resnet", "residual", "skip connection"))

    if has_transformer:
        blocks.append(
            "### 🗺️ 1. Transformer Architecture & Residual Stream Diagram\n"
            "```text\n"
            "  Input Tokens (x₁, …, xₙ)\n"
            "         │\n"
            "         ▼\n"
            "  ┌───────────────────────────────────────────────┐\n"
            "  │ Token Embedding (ℝⁿˣᵈ) + Positional Encoding  │\n"
            "  └──────────────────────┬────────────────────────┘\n"
            "                         │  r⁽⁰⁾ ∈ ℝ^[B × n × d_model]\n"
            "         ╔═══════════════▼════════════════════════════════════════╗\n"
            "         ║  Transformer Block (× N Layers)                        ║\n"
            "         ║                                                        ║\n"
            "         ║   r⁽ᵏ⁾ ──┬──► LayerNorm ──► Multi-Head Attention (Q,K,V)║\n"
            "         ║          │                 Attention = softmax(QKᵀ/√dₖ)V║\n"
            "         ║          └──► ( + ) ◄────── Wᴼ Concat(head₁ … headₕ)   ║\n"
            "         ║                 │                                      ║\n"
            "         ║   r'⁽ᵏ⁾ ─┬──► LayerNorm ──► Feed-Forward Network (FFN) ║\n"
            "         ║          │                 FFN(z) = GELU(zW₁ + b₁)W₂+b₂║\n"
            "         ║          └──► ( + ) ◄────── Residual Add               ║\n"
            "         ╚═════════════════╤══════════════════════════════════════╝\n"
            "                           │  H ∈ ℝ^[B × n × d_model]\n"
            "                           ▼\n"
            "               Linear Head + Softmax ──► Output Probabilities\n"
            "```"
        )

    if has_cnn_ops:
        blocks.append(
            "### 🗺️ 2. Convolution (Stride & Padding) & Pooling Spatial Diagram\n"
            "```text\n"
            "  PADDING (P=1 Zero-Border)          STRIDE (S=2 Step Jump)           POOLING (2×2 Max / Avg)\n"
            "  ┌───┬───┬───┬───┬───┐              ┌───┬───┬───┬───┐                ┌───────┬───────┐\n"
            "  │ 0 │ 0 │ 0 │ 0 │ 0 │              │ K │ K │ . │ . │ ──(S=2)──►     │ 1   3 │ 2   4 │      ┌───┬───┐\n"
            "  ├───┼───┼───┼───┼───┤              ├───┼───┼───┼───┤                │ 5   8 │ 1   6 │ ───► │ 8 │ 6 │ (MaxPool)\n"
            "  │ 0 │ x │ x │ x │ 0 │              │ K │ K │ . │ . │                ├───────┼───────┤      ├───┼───┤\n"
            "  ├───┼───┼───┼───┼───┤              └───┴───┴───┴───┘                │ 2   4 │ 9   3 │      │ 7 │ 9 │\n"
            "  │ 0 │ x │ x │ x │ 0 │              Spatial Output Dimension:        │ 7   1 │ 0   2 │      └───┴───┘\n"
            "  └───┴───┴───┴───┴───┘              H_out = ⌊(H + 2P - K) / S⌋ + 1   └───────┴───────┘      Halves H×W (S=2)\n"
            "```"
        )

    if has_resnet:
        blocks.append(
            "### 🗺️ 3. ResNet Residual Block (Identity Skip Connection) Diagram\n"
            "```text\n"
            "                  Input Tensor  x\n"
            "                        │\n"
            "         ┌──────────────┴──────────────┐\n"
            "         │                             │ (Identity Shortcut / Skip Connection)\n"
            "         ▼                             │\n"
            "  ┌──────────────────────────┐         │\n"
            "  │ Conv 3×3 ──► BatchNorm   │         │\n"
            "  │ ──► ReLU ──► Conv 3×3    │  F(x)   │  x\n"
            "  │ ──► BatchNorm            │         │\n"
            "  └──────────────┬───────────┘         │\n"
            "                 │                     │\n"
            "                 └──────► ( + ) ◄──────┘\n"
            "                            │\n"
            "                            ▼\n"
            "                 y = ReLU( F(x, {Wᵢ}) + x )\n"
            "       Gradient Flow: ∂L/∂x = ∂L/∂y · (1 + ∂F/∂x)  [Prevents Vanishing Gradients]\n"
            "```"
        )

    if not blocks:
        return ""

    return "\n\n## 🗺️ Visual Architecture & Tensor-Flow Diagrams\n\n" + "\n\n".join(blocks) + "\n\n"


def format_formulas_and_diagrams(report: str, query: str, include_diagrams: bool = True) -> str:
    """
    Post-processes the synthesized Markdown report so that:
    1. Raw/jammed LaTeX strings (`$$...$$,$$...$$` or `\\[...\\]`) are converted into clean,
       properly formatted Unicode formula blocks that never break in Markdown parsers.
    2. Bare LaTeX macros (`\\text{}`, `\\mathbf{}`, `\\frac{}{}`, `\\qquad`, etc.) anywhere in lines
       are converted to clean Unicode math.
    3. Visual Architecture & Tensor-Flow Diagrams are added for open-ended architectural queries
       (disabled when synthesizing strictly from retrieved evidence so unretrieved diagram labels
       cannot degrade factual grounding).
    """
    if not report or not report.strip():
        return report

    text = report

    # 0. Protect Markdown tables: if `$$...$$` or `\[...\]` appears inside a table row (line has `|`),
    #    convert it to a single-line inline `code` span so it NEVER breaks the table row with newlines/fences.
    def _replace_table_display_math(m: re.Match) -> str:
        prefix = m.group(1)
        inner = m.group(2).strip()
        if not inner:
            return prefix
        parts = [latex_to_readable_math(p).replace("|", "∣") for p in re.split(r"\\qquad|\n+", inner) if p.strip()]
        cleaned = " ; ".join(parts)
        return f"{prefix}`{cleaned}`"

    for _ in range(4):
        text = re.sub(r"(?m)^([^\n]*\|[^\n]*?)\$\$([\s\S]*?)\$\$[,;.]?", _replace_table_display_math, text)
        text = re.sub(r"(?m)^([^\n]*\|[^\n]*?)\\\[([\s\S]*?)\\\][,;.]?", _replace_table_display_math, text)

    # 1. Split comma-separated display math blocks outside tables: `$$...$$,$$...$$` -> separate lines
    text = re.sub(r"\$\$\s*[,;]\s*\$\$", "$$\n\n$$", text)

    # 2. Convert standalone display math blocks `$$...$$` and `\[...\]` into clean formatted Formula Callout blocks
    def _replace_display_math(m: re.Match) -> str:
        inner = m.group(1).strip()
        if not inner:
            return ""
        parts = [p.strip() for p in re.split(r"\\qquad|\n+", inner) if p.strip()]
        readable_lines = [f"  • {latex_to_readable_math(p)}" for p in parts]
        joined = "\n".join(readable_lines)
        return f"\n\n```text\n📐 Mathematical Formulation:\n{joined}\n```\n\n"

    text = re.sub(r"\$\$([\s\S]*?)\$\$[,;.]?", _replace_display_math, text)
    text = re.sub(r"\\\[([\s\S]*?)\\\][,;.]?", _replace_display_math, text)

    # 3. Convert inline math `\(...\)` and `$...$` (excluding currency like $100) into clean inline code `...`
    def _replace_inline_math(m: re.Match) -> str:
        inner = m.group(1).strip()
        if not inner:
            return ""
        cleaned = latex_to_readable_math(inner).replace("|", "∣")
        return f"`{cleaned}`"

    text = re.sub(r"\\\(([^\n]*?)\\\)", _replace_inline_math, text)
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]{1,120}?)(?<!\$)\$(?!\$)", _replace_inline_math, text)

    # 4. Clean any remaining bare LaTeX macros on lines where $$ was omitted or stripped
    _BARE_MACROS = (
        r"\text{", r"\mathbf{", r"\frac{", r"\qquad", r"\lfloor", r"\sum", r"\mathbb{",
        r"\cdot", r"\times", r"\in", r"\Delta", r"\delta", r"\phi", r"\varphi", r"\Phi",
        r"\theta", r"\Theta", r"\sigma", r"\Sigma", r"\epsilon", r"\varepsilon", r"\alpha",
        r"\beta", r"\gamma", r"\lambda", r"\mu", r"\sim", r"\hat", r"\approx", r"\leq", r"\geq",
    )
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        if any(macro in line for macro in _BARE_MACROS):
            if "|" in line:
                # Clean each Markdown table cell individually so `|` column separators stay intact
                cells = line.split("|")
                cleaned_cells = [
                    f" {latex_to_readable_math(c)} " if any(m in c for m in _BARE_MACROS) else c
                    for c in cells
                ]
                cleaned_lines.append("|".join(cleaned_cells))
            else:
                cleaned_lines.append(latex_to_readable_math(line))
        else:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # 5. Inject visual Architecture & Tensor-Flow Diagrams only when allowed and not already present
    has_diagram = "┌──" in text or "╔══" in text or "```mermaid" in text
    if include_diagrams and not has_diagram:
        diagrams_md = build_architecture_diagrams(query)
        if diagrams_md:
            if "## ⚙️" in text:
                idx = text.index("## ⚙️")
                text = text[:idx].rstrip() + diagrams_md + text[idx:]
            elif "## 📊" in text:
                idx = text.index("## 📊")
                text = text[:idx].rstrip() + diagrams_md + text[idx:]
            else:
                text = text.rstrip() + diagrams_md

    return text
