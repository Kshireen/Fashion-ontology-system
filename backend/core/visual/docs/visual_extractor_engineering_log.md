# Fashion Ontology System

## Visual Feature Extraction Module – Engineering Log

**Author:** Ruslan Khan

**Branch:** `feature/visual-extractor`

---

# Goal

Extend the existing ontology-based fashion understanding system with a computer vision pipeline capable of extracting semantic fashion attributes directly from product images.

The long-term objective is to build a **multimodal ontology engine** where textual and visual information are mapped into the same ontology.

---

# Phase 1 – Research

## Problem Statement

The original ontology engine relied entirely on textual metadata.

Example:

```
Product Name
Description
Meta Information
```

This approach fails when:

* Product descriptions are incomplete.
* Important visual characteristics are missing.
* Sellers provide poor metadata.
* Images contain information unavailable in text.

Example:

Text:

```
Women's Shirt
```

Image:

* Floral
* Long Sleeve
* Blue
* Cotton

Most useful information exists only in the image.

---

# Phase 2 – Technology Exploration

Several vision approaches were considered.

### Option 1 – OpenCV

Capabilities

* Dominant colors
* Edge detection
* Texture analysis
* Pattern heuristics

Advantages

* Fast
* CPU friendly

Disadvantages

* Rule based
* Difficult to generalize

Decision

Use only for classical preprocessing.

---

### Option 2 – CNN Feature Extractor

Candidate

```
ResNet50
```

Advantages

* Pretrained
* Good feature embeddings

Disadvantages

* Not trained for semantic fashion understanding
* Requires additional classifiers

Decision

Rejected.

---

### Option 3 – CLIP

Candidate

```
OpenAI CLIP
```

Advantages

* Zero-shot recognition
* Natural language interface
* Excellent semantic embeddings
* No supervised training required

Decision

Selected.

---

# Phase 3 – Architecture

A dedicated visual module was introduced.

```
core/

    visual/

        clip_encoder.py
        preprocessing.py
        visual_result.py
        test_clip_encoder.py
```

Reason

Keep vision completely independent from ontology logic.

---

# Phase 4 – Image Preprocessing

Created

```
preprocessing.py
```

Responsibilities

* Image loading
* RGB conversion
* Resize
* Tensor preparation
* CPU compatible pipeline

Design Goal

Every image entering CLIP follows the exact same preprocessing pipeline.

---

# Phase 5 – CLIP Encoder

Created

```
clip_encoder.py
```

Responsibilities

* Load CLIP model
* CPU inference
* Generate image embeddings
* Generate text embeddings
* Cosine similarity
* Zero-shot classification

Pipeline

```
Image

↓

CLIP Encoder

↓

Image Embedding

↓

Similarity

↓

Fashion Labels
```

---

# Phase 6 – Zero-Shot Classification

Instead of training a classifier, candidate labels are supplied directly.

Example

```
floral shirt
striped shirt
plain shirt
denim jacket
```

CLIP calculates semantic similarity.

Example Output

```
floral shirt      0.244
plain shirt       0.224
denim jacket      0.126
striped shirt     0.123

Prediction

floral shirt
```

Outcome

Zero-shot pipeline successfully validated.

---

# Phase 7 – Result Object

Created

```
visual_result.py
```

Purpose

Provide a structured prediction object instead of returning raw tensors.

Contains

* label
* confidence
* similarity score
* ranked predictions

Advantages

* Cleaner API
* Easier debugging
* Future extensibility

---

# Phase 8 – Testing

Created

```
test_clip_encoder.py
```

Tested

* Model loading
* CPU inference
* Embedding generation
* Similarity computation
* Zero-shot prediction

Verified Output

```
floral shirt
```

Result

Pipeline working correctly on CPU.

---

# Phase 9 – Engineering Challenges

## Module Import Issues

Problem

```
ModuleNotFoundError
```

Cause

Running test files directly from nested folders.

Solution

Run modules using project root.

---

## CPU Compatibility

Requirement

No GPU dependency.

Solution

Model automatically selects

```
torch.device("cpu")
```

Pipeline tested successfully.

---

## HuggingFace Download Warning

Observed

```
Unauthenticated requests
```

Analysis

Not an error.

Impact

Model downloads successfully.

Optional Improvement

Configure HF_TOKEN for faster downloads.

---

# Current Architecture

```
Image

↓

Preprocessing

↓

CLIP Encoder

↓

Image Embedding

↓

Text Embedding

↓

Cosine Similarity

↓

Zero-Shot Labels
```

---

# Current Status

Completed

✓ Image preprocessing

✓ CLIP encoder

✓ Zero-shot inference

✓ CPU execution

✓ Result abstraction

✓ Unit testing

Pending

* Connect CLIP predictions with lexical layer
* Normalize visual labels
* Map visual labels into ontology concepts
* Merge textual and visual features
* Store multimodal product representation

---

# Next Milestone

Integrate the visual pipeline into the ontology engine.

Target Architecture

```
               Product
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
 Text Pipeline          Visual Pipeline
      │                       │
      ▼                       ▼
Lexical Terms         CLIP Predictions
      │                       │
      └───────────┬───────────┘
                  ▼
          Lexical Normalization
                  ▼
           Concept Mapping
                  ▼
          Product Ontology
```

---

# Summary

The visual module has been successfully implemented as an independent component capable of extracting semantic fashion attributes using OpenAI CLIP. The pipeline is fully CPU compatible, supports zero-shot classification, and is intentionally decoupled from the ontology layer to maintain modularity. The next stage is multimodal integration, where CLIP predictions will be normalized through the lexical layer and mapped into ontology concepts alongside textual features.
