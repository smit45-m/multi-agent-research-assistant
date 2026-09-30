

## Slide 1

# So what is Jev?

It's a new AI model.


## Slide 2

## So what? A lot of models come up every other week.

## But Jev is different.

# It's not an LLM.

---

*(Faint background text visible in rows:)*
- Halcyon-3 Numina 70B Corvid Mini Tessellate-XL Mar1
- Obsidian-40E Vesper-Turbo Verdigris 8x22 Lacuna-1.5
- Sombre 32K Quillon Air Fathom-9 Basalt Ultra Nimbus

*(Footer information)*
2 / 16


## Slide 3

QUESTION
Is this ticket urgent?

LLM
one token at a time

token 10

ANSWER
Yes, this ticket seems urgent because the customer ...

# Jev does not generate text.

Illustrative token stream — not model output.

3 / 16


## Slide 4

## So what is Jev?

Jev is an AI model built to make fast, structured decisions that software can use directly.


## Slide 5

## STATE
"My package arrived damaged and I want a refund."

## QUESTION
Which queue should handle this?

## OPTIONS
- billing
- shipping
- technical
- general

# Jev
one parallel pass

~120 ms

## ANSWER
- shipping: 0.71
- billing: 0.24
- general: 0.04
- technical: 0.01

No text. No parsing. Just a typed answer with a probability.

Illustrative output — not a live model call.

5 / 16


## Slide 6

# So what? Any LLM can do this with structured output.

True. It can. Two things change.

- 01 Speed
- 02 Cost

Both measured on the same job: triaging a support inbox with Jev and with an LLM.

6 / 16


## Slide 8

LIVE

# Second difference: cost.

Per million tokens — in: LLM ₹119.50 • Jev ₹4.02 out: LLM ₹956.00 • Jev free

Same 1,000-token input for both • 10,000 emails a day.

| | LLM | Jev |
| :--- | :--- | :--- |
| Per email | ₹0.326 | ₹0.004 |
| Per day | ₹3,260 | ₹40 |
| Per year | ₹11,89,885 | ₹14,655 |

## 81.2× cheaper — ₹11,75,230 a year saved

Vendor headline 444.6× on their evals; here 81.2×, almost all of it the LLM's reasoning tokens.

Figures are TypeSafe's own, from their launch post (Sep 2026). Independent benchmarks are still limited. LLM list prices converted at ₹95.6 / USD (23 Sep 2026).

8 / 16


## Slide 9

# Other benefits
9.1 Many questions, one pass  9.2 Confidence you can act on  9.3 It can't make things up

Shared axis · full width = 7.0 s

**Jev — 5 questions** ≈ 130 ms

**LLM — one call per question** ≈ 7.0 s

# Adding a question adds almost no time or cost.

Illustrative output — not a live model call.
9.1 / 16


## Slide 10

## Other benefits

9.1 Many questions, one pass  9.2 Confidence you can act on  9.3 It can't make things up

## Trained so that 90% confident means right about 90% of the time.

- `x: stated confidence · y: actual accuracy`
- `- target` against the dashed ideal diagonal

Calibration is Jev's training objective (RLCD). TypeSafe has not yet published a calibration curve.

Illustrative output — not a live model call.

9.2 / 16


## Slide 12

Behind Jev
Who
Why
Impact
Where

# Who built Jev?

DA

# Diogo Almeida
ex-OpenAI • InstructGPT • ChatGPT

## Now building AI for software, not people.

TypeSafe AI

10.1 / 16


## Slide 13

Behind Jev

Who
Why (highlighted in lime green)
Impact
Where

Software needs System 1 thinking, but we keep building System 2 thinking.

10.2 / 16


## Slide 14

Behind Jev

Who
Why
Impact
Where

AI moves from a feature to a primitive.

```python
if jev("Is this customer angry?") > 0.9:
```

Illustrative

AI stops being the product. It becomes plumbing.

10.3 / 16


## Slide 15

# Behind Jev

Who | Why | Impact | Where 1 / 5
--- | --- | --- | ---

# 01

# AI agents

Which tool? · Which model? · Is this step safe?

## Every small decision, fast and cheap enough to check.

10.4 / 16


## Slide 17

# Behind Jev
Who | Why | Impact | Where 3 / 5

## 03
# Trust & safety

Moderation • Fraud • Spam • Jailbreak detection

## Score, threshold, escalate, on every single request.

10.6 / 16
