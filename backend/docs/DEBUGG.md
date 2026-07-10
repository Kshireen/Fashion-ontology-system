# Fashion Ontology Engine — Debugging & Testing Log

A record of the issues found, root causes, and fixes made while getting the Fashion Ontology
Engine demo-ready. Useful both as an interview reference and as documentation of real
engineering judgment under time pressure.

---

## 1. Environment Setup Issues

### Torch/torchvision version incompatibility
**Problem:** `pip install -r requirements.txt` failed — `torch==2.1.0` had no wheel for Python 3.12
(torch added 3.12 support only from 2.2.0 onward).

**Fix:** Relaxed pin to `torch>=2.2.0`, later installed CPU-only build via
`--index-url https://download.pytorch.org/whl/cpu` to avoid a multi-GB GPU download.

### SSL errors during large package downloads
**Problem:** `SSLError: DECRYPTION_FAILED_OR_BAD_RECORD_MAC` repeatedly interrupted downloads of
large packages (torch, opencv, scipy) — a network-layer issue, not a code or dependency problem.

**Fix:** Switched to a mobile hotspot; install completed without further SSL errors.

**Takeaway:** Confirmed the failure was environmental by noting it persisted even after commenting
out torch entirely — the same error hit a different (also large) package next in the resolver.

---

## 2. Django Routing Issues

### Root URLconf missing the app's routes
**Problem:** All `/api/...` endpoints returned 404 despite the server running — `config/urls.py`
only registered `admin/`, never included the app-level `urls.py`.

**Fix:**
```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('api.urls')),
]
```

### DRF browsable API template missing
**Problem:** `/api/ontology/` returned 500: `TemplateDoesNotExist: rest_framework/api.html`.

**Root cause:** `rest_framework` app was never added to `INSTALLED_APPS`, so DRF's own templates
weren't discoverable.

**Fix:** Added `'rest_framework'` to `INSTALLED_APPS`.

---

## 3. Data Pipeline Bug: Pandas NaN Truthiness

**Problem:** Batch-processing the real Shirts catalog CSV produced zero extracted features for
every row, despite the same text working fine when tested manually against the lexical layer.

**Root cause:** The `description` column was empty for most rows; a fallback was written as:
```python
str(row.get('description') or row.get('meta_info', ''))
```
Pandas represents empty cells as `NaN` (a float), and **`NaN` is truthy in Python** — so
`NaN or meta_info` evaluated to `NaN`, never falling through to `meta_info`. The result:
`description` became the literal string `"nan"`, which matched nothing.

**Fix:**
```python
desc = row.get('description')
if pd.isna(desc) or str(desc).strip().lower() in ('', 'nan'):
    desc = row.get('meta_info', '')
if pd.isna(desc):
    desc = ''
```

**Why this is worth mentioning in an interview:** it's a subtle, easy-to-miss gotcha
(`NaN or X` ≠ `"" or X`) that silently produces "successful" results with zero useful output —
no exception, no error, just empty features across the board.

---

## 4. Ontology Bug: Token-Level vs. Phrase-Level Ambiguity

**Problem:** After the NaN fix, features started extracting — but incorrectly for several
products. A shirt listed as "Short Sleeve" resolved to `length_mini` (a garment-length concept),
and "Loose Fit" resolved to `fit_oversized`.

**Root cause:** The lexical vocabulary held single-word aliases with real-world ambiguity:
- `"mini"` was aliased to `"short"` — correct in the context of *garment length* ("mini skirt"),
  wrong in the context of *sleeve length* ("short sleeve").
- `"oversized"` was aliased to `"loose"` — correct for casual fit descriptions in some contexts,
  but "Loose Fit" is a distinct, standard fit category of its own, not synonymous with oversized.

The matcher's n-gram logic (3-word → 2-word → 1-word) was already phrase-first, but the
**vocabulary itself only defined single-word entries** for these terms, so multi-word phrases
fell through to the ambiguous single-word fallback.

**Fix:** Added explicit multi-word canonical terms —
`short_sleeve`, `long_sleeve`, `three_quarter_sleeve`, `loose_fit`, `regular_fit`, `oversized_fit` —
and removed the ambiguous single-word aliases (`"short"` no longer aliases to `"mini"`) so that
standalone occurrences of ambiguous words no longer resolve to anything, rather than resolving
to something wrong.

**Verification:**
```
Input:  "Short Sleeve Loose Fit Shirt ... with Long Sleeve option"
Before: ['mini', 'oversized', ...]         ← wrong
After:  ['short_sleeve', 'loose_fit', 'long_sleeve']   ← correct
```

**Why this matters architecturally:** this is a real instance of why fashion attribute
extraction can't be pure keyword matching — meaning is phrase-dependent, not word-dependent.
The proper long-term fix (discussed but not fully implemented under time pressure) is full
phrase-first canonicalization across the entire vocabulary, not just the terms that happened to
surface as bugs during this test run.

---

## 5. Multimodal (CLIP) Visual Extraction — Tested Standalone

Built and tested a CLIP-based zero-shot image classifier (`core/visual/clip_encoder.py`,
`zero_shot_classifier.py`) as a second modality alongside the text-based ontology.

**Test 1 — Zero-shot classifier**, run against a real catalog product image (a floral SHEIN
shirt already resolved via text as `pattern_floral`):
```
Prediction(label='plain shirt', score=0.224)
Prediction(label='paisley pattern', score=0.211)
Prediction(label='floral pattern', score=0.201)
```

**Test 2 — Direct CLIP cosine similarity** against candidate labels, same image:
```
floral shirt         0.244   ← highest
striped shirt        0.123
plain shirt          0.224
denim jacket         0.126
Prediction: floral shirt
```

**Result:** CLIP independently predicted "floral" as the top or near-top match — corroborating
the text-based lexical resolution without any trained classifier, using zero-shot text-prompt
similarity only.

**Known gap (stated honestly, not hidden):** this visual pipeline is **not yet wired into**
`MultimodalFeatureExtractor` / the Concept-Instance layers. It exists and is tested as a
standalone module on the `visual-exctractor` branch. The next integration step is mapping CLIP's
top-label predictions onto existing ontology concepts (e.g. "floral shirt" → `pattern_floral`)
so visual signal contributes to the same resolution trace as text does.

---

## 6. Git Hygiene Fixes

- Added `.gitignore` entries for `venv/`, `__pycache__/`, `*.pyc`, `db.sqlite3`, `.env` — these
  were previously untracked but at risk of being committed.
- Removed accidentally-committed `__pycache__/*.pyc` files and test-run `outputs/*.json`
  artifacts from the `visual-exctractor` branch.
- Kept work on three branches deliberately separate rather than force-merging everything into
  `main`:
  - `main` — stable, tested three-layer ontology + phrase-based lexical fix
  - `feature/phrase-based-lexical-matching` — merged into `main` after verification
  - `visual-exctractor` — kept separate; CLIP work is tested standalone but not yet integrated

---

## Summary 

The through-line across all of these: every fix came from **actually running the system against
real catalog data** rather than trusting it in the abstract. The NaN bug, the phrase-ambiguity
bug, and the routing/config issues would not have surfaced without end-to-end testing. The CLIP
work is honestly scoped as tested-but-unintegrated rather than overstated as "done."