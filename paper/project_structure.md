# MSc Project Notes

## Project Direction

**Topic:** Audio-visual deepfake detection
**Focus:** Modality-specific shortcut learning, multimodal fusion, robustness, and failure analysis.

This project should not simply be about building a deepfake detector. The research should investigate whether multimodal deepfake detectors genuinely use both audio and visual evidence, or whether high overall accuracy hides failures on specific manipulation types.

---

## Working Titles

* **Modality-Specific Shortcut Learning in Audio-Visual Deepfake Detection**
* **Do Multimodal Deepfake Detectors Generalise? A Robustness and Modality-Mismatch Study**
* **Beyond Accuracy: Diagnosing Modality Bias in Audio-Visual Deepfake Detection**

---

## Core Research Aim

To investigate whether audio-visual deepfake detectors genuinely benefit from multimodal evidence, or whether they rely on shortcuts from one dominant modality and fail under modality mismatch, imbalance, or degraded conditions.

---

## Main Research Question

> Do multimodal deepfake detectors actually use both audio and visual evidence, or can high aggregate accuracy hide failures on fake-audio-only, fake-video-only, or degraded-modality cases?

---

## Research Gap

Many deepfake detection systems are evaluated using overall real/fake accuracy. This can be misleading for multimodal deepfake detection because a model may perform well overall while failing on specific manipulation types.

For example, a model may:

* perform well when the video is fake,
* fail when only the audio is fake,
* over-rely on visual cues,
* ignore the audio stream,
* appear strong because the dataset is imbalanced.

The project should argue that multimodal deepfake detection needs **modality-aware evaluation**, not just overall accuracy.

---

## Main Dataset

### FakeAVCeleb

Dataset categories:

| Category | Meaning             |
| -------- | ------------------- |
| A        | RealVideo-RealAudio |
| B        | RealVideo-FakeAudio |
| C        | FakeVideo-RealAudio |
| D        | FakeVideo-FakeAudio |

Useful labels to create:

```text
video_fake = category in C or D
audio_fake = category in B or D
overall_fake = category is not A
```

---

## Initial Dataset Concern

The dataset appears imbalanced:

```text
A: RealVideo-RealAudio      500
B: RealVideo-FakeAudio      500
C: FakeVideo-RealAudio      ~9700
D: FakeVideo-FakeAudio      ~10800
```

This imbalance is not just a problem. It can become part of the research.

Potential claim:

> Natural training on imbalanced multimodal deepfake data may bias models toward detecting visual manipulation while underperforming on fake-audio-only samples.

---

## Research Hypotheses

### H1 — Overall Accuracy Is Misleading

Overall real/fake accuracy will overestimate model reliability because performance differs significantly across fake-audio-only, fake-video-only, fake-both, and real-real categories.

### H2 — Dataset Imbalance Causes Modality Bias

Models trained on the natural imbalanced dataset distribution will be biased toward visual fake detection and perform worse on fake-audio-only samples.

### H3 — Naive Fusion May Hide Failures

Multimodal fusion may improve overall accuracy while still failing on specific modality conditions.

### H4 — Balanced Training Helps Minority Conditions

Balanced training across A/B/C/D categories may improve fake-audio-only and fake-video-only performance, even if overall accuracy changes.

### H5 — Modality Dropout May Improve Robustness

Training with modality dropout may reduce over-reliance on one modality and improve robustness when audio or video is missing/degraded.

### H6 — Larger Models Are Not Automatically More Robust

Larger encoders may improve clean accuracy but may not solve modality bias without appropriate training and evaluation.

---

## Proposed Contribution

This project proposes a diagnostic evaluation framework for audio-visual deepfake detection.

Instead of evaluating only overall real/fake accuracy, it separately measures performance on:

* real video + real audio,
* real video + fake audio,
* fake video + real audio,
* fake video + fake audio.

It investigates:

1. whether multimodal fusion improves performance across all manipulation types,
2. whether dataset imbalance encourages modality-specific shortcut learning,
3. whether the model over-relies on video or audio,
4. whether balanced training or modality dropout improves robustness,
5. whether larger/foundation models improve genuine robustness or only clean accuracy.

---

## What Would Make This Weak

Avoid making the project only:

```text
Train visual model.
Train audio model.
Concatenate features.
Report overall accuracy.
```

That is too implementation-focused and not enough for the research ambition.

---

## What Would Make This Strong

The project becomes strong if it includes:

```text
Modality-aware evaluation
Balanced vs imbalanced training
Fusion method ablation
Missing/degraded modality tests
Failure analysis
Identity-disjoint splits
Clear comparison between small and larger models
```

---

# Work Packages

## WP1 — Literature Review

### Topics to Review

* Visual deepfake detection
* Audio deepfake detection
* Audio-visual/multimodal deepfake detection
* Deepfake datasets
* Dataset bias and shortcut learning
* Cross-dataset generalisation
* Robustness under degradation
* Fusion methods
* Modality dropout / missing modality robustness
* Foundation models for vision/audio

### Notes Template for Each Paper

```md
## Paper Title

**Citation:**  
**Problem:**  
**Method:**  
**Dataset:**  
**Main Result:**  
**Limitation:**  
**Relevance to my project:**  
**Useful quote/idea:**  
```

### Literature Review Goal

The literature review should justify why overall accuracy is insufficient and why modality-aware evaluation is needed.

---

## WP2 — Dataset Audit

### Tasks

* [ ] Load FakeAVCeleb metadata CSV.
* [ ] Count samples by A/B/C/D category.
* [ ] Count samples by generation method.
* [ ] Count samples by identity/source.
* [ ] Count samples by race/gender metadata.
* [ ] Check missing or corrupt files.
* [ ] Check whether audio exists for all videos.
* [ ] Check video duration statistics.
* [ ] Check video resolution statistics.
* [ ] Check audio sample rate statistics.
* [ ] Verify paths in metadata match actual files.
* [ ] Create cleaned manifest CSV.

### Clean Manifest Columns

```text
sample_id
filename
full_path
category
type
video_fake
audio_fake
overall_fake
method
source_id
target1_id
target2_id
gender
race
split
```

### Important Split Rule

Do not randomly split rows.

Use identity-disjoint splits where possible:

```text
train identities ≠ validation identities ≠ test identities
```

This avoids leakage and makes results more credible.

---

## WP3 — Preprocessing Pipeline

### Visual Pipeline

```text
video
→ sample frames
→ detect/crop face
→ resize
→ normalise
→ save crops or load dynamically
```

Initial settings:

```text
8–16 frames per video
224x224 face crops
```

Stretch settings:

```text
16–32 frames per video
higher resolution if GPU allows
```

### Audio Pipeline

```text
video/audio file
→ extract audio
→ resample
→ convert to mono
→ compute log-mel spectrogram/MFCC
→ train audio model or extract embeddings
```

Initial audio approach:

```text
log-mel spectrogram + CNN
```

Stretch audio approach:

```text
wav2vec2 / Whisper embeddings + classifier
```

---

# Models

## Model A — Visual-Only Detector

Input:

```text
face crops / sampled frames
```

Possible models:

```text
EfficientNet-B0/B3
Xception
CLIP/ViT as stretch
```

Target:

```text
video_fake
```

Purpose:

To measure whether visual evidence alone can detect fake video manipulation.

---

## Model B — Audio-Only Detector

Input:

```text
log-mel spectrograms / audio embeddings
```

Possible models:

```text
small CNN
wav2vec2 embeddings
Whisper embeddings as stretch
```

Target:

```text
audio_fake
```

Purpose:

To measure whether audio evidence alone can detect fake audio manipulation.

---

## Model C — Late Fusion

Input:

```text
visual probability + audio probability
```

Fusion methods:

```text
average
weighted average
simple logistic regression
```

Purpose:

To test whether combining unimodal predictions improves detection.

---

## Model D — Learned Fusion

Input:

```text
visual embedding + audio embedding
```

Model:

```text
MLP classifier
```

Purpose:

To test whether learned multimodal fusion outperforms simple late fusion.

---

## Model E — Modality Dropout Fusion

Training idea:

Randomly weaken or remove one modality during training.

Examples:

```text
drop audio
blur video
add audio noise
drop video frames
mask audio features
```

Purpose:

To force the model not to over-rely on one modality.

---

## Model F — Large/Foundation Model Comparison

Only after the basic pipeline works.

Possible visual models:

```text
CLIP ViT-B/16
ViT-L
DINOv2
Swin Transformer
```

Possible audio models:

```text
wav2vec2
Whisper embeddings
```

Purpose:

To test whether model scale improves genuine robustness or only clean accuracy.

---

# Experiments

## Experiment 1 — Sanity Check

Goal:

Confirm that the dataset, labels and preprocessing are correct.

Tasks:

* [ ] Train tiny visual model on small subset.
* [ ] Train tiny audio model on small subset.
* [ ] Confirm both beat random performance.
* [ ] Confirm labels are correctly mapped.
* [ ] Confirm video-level evaluation works.

If this fails, stop and fix the pipeline.

---

## Experiment 2 — Single-Modality Baselines

Models:

```text
visual-only
audio-only
```

Evaluate on:

```text
A: RealVideo-RealAudio
B: RealVideo-FakeAudio
C: FakeVideo-RealAudio
D: FakeVideo-FakeAudio
```

Expected behaviour:

```text
visual-only should struggle on B
audio-only should struggle on C
```

This is not a bad result. It proves why multimodal evaluation matters.

---

## Experiment 3 — Multimodal Fusion

Compare:

```text
visual-only
audio-only
late fusion
learned fusion
```

Main table:

```text
Model | A Real/Real | B AudioFake | C VideoFake | D BothFake | Overall
```

Key question:

Does fusion improve all categories, or only the easiest ones?

---

## Experiment 4 — Natural vs Balanced Training

Train two versions:

```text
1. Natural imbalanced dataset distribution
2. Balanced A/B/C/D subset
```

Compare per-category F1.

Key question:

Does natural training produce a video-dominant model that performs badly on fake-audio-only cases?

Possible useful finding:

Balanced training may reduce overall accuracy but improve minority modality robustness.

---

## Experiment 5 — Fusion Ablation

Compare:

```text
visual-only
audio-only
late fusion
learned fusion
learned fusion + balanced training
learned fusion + modality dropout
```

Goal:

Identify which fusion strategy is most robust across modality conditions.

---

## Experiment 6 — Missing/Degraded Modality Robustness

Test:

```text
mute audio
replace audio with noise
blur video
compress video
drop frames
reduce video quality
```

Questions:

* Does the multimodal model collapse when one modality is degraded?
* Does it rely mostly on video?
* Does audio help when video is degraded?
* Does modality dropout improve robustness?

---

## Experiment 7 — Failure Analysis

Analyse:

```text
false positives
false negatives
high-confidence wrong predictions
fake-audio-only samples missed by multimodal model
fake-video-only samples missed by multimodal model
degraded samples where model collapses
```

Save examples into:

```text
results/failure_cases/
```

Questions:

* What does the model fail on?
* Is one modality ignored?
* Are some manipulation methods easier?
* Does dataset imbalance explain failures?
* Are failures concentrated in specific identities, methods, or categories?

---

## Experiment 8 — Model Scaling

Only if time/GPU allows.

Compare:

```text
EfficientNet/Xception baseline
CLIP/ViT-B visual encoder
larger ViT/foundation model
```

Question:

Does model scale improve robustness, or only clean/in-distribution performance?

Possible important result:

A larger model may improve overall performance but still fail on fake-audio-only or degraded-modality cases.

---

# Evaluation Metrics

Use:

```text
accuracy
precision
recall
F1
ROC-AUC
confusion matrix
per-category F1
performance drop under degradation
video-level metrics
```

Do not rely only on frame-level accuracy.

Important metric style:

```text
clean F1
degraded F1
performance drop = clean F1 - degraded F1
```

---

# Key Tables and Figures

## Table 1 — Dataset Summary

```text
Category | Type | Count | Video Fake? | Audio Fake?
```

## Table 2 — Method Distribution

```text
Generation Method | Count | Category
```

## Table 3 — Main Results

```text
Model | A | B | C | D | Overall
```

## Table 4 — Natural vs Balanced Training

```text
Training Setup | A F1 | B F1 | C F1 | D F1 | Overall F1
```

## Table 5 — Fusion Ablation

```text
Model | A | B | C | D | Overall | Notes
```

## Table 6 — Robustness Results

```text
Model | Clean | Audio Noise | Muted Audio | Blurred Video | Compressed Video
```

## Figure Ideas

* Dataset imbalance bar chart
* Confusion matrices per model
* Per-category F1 bar chart
* Robustness drop chart
* Fusion comparison chart
* Failure case examples

---

# Supervisor Questions

Ask:

1. Is this narrowed multimodal direction suitable?
2. Is FakeAVCeleb enough as the core dataset?
3. Should I include FaceForensics++, Celeb-DF, DeeperForensics, or Fake-Or-Real as supporting datasets?
4. Should I prioritise modality bias/fusion analysis over trying to achieve SOTA accuracy?
5. Is balanced vs imbalanced training a strong enough research angle?
6. Would modality dropout/gated fusion be a suitable contribution?
7. What would make this strong enough to potentially develop into a workshop paper?
8. Can I get GPU/storage access for repeated experiments?
9. Should I request 24GB GPUs or try to access A100-class GPUs for the model-scaling component?

---

# GPU Justification

## Minimum Requirement

```text
16GB CUDA GPU
```

Enough for:

```text
EfficientNet/Xception
small audio CNN
sampled frames
basic fusion
```

## Preferred Requirement

```text
24GB CUDA GPU
```

Enough for:

```text
larger batch sizes
more sampled frames
larger CNN/ViT-B encoders
repeated ablations
more stable training
```

## Ideal Requirement

```text
A100 or similar high-memory GPU
```

Justification:

The core project can be completed with 16–24GB VRAM using smaller models. However, A100-class access would support the more ambitious version of the project by enabling:

```text
larger pretrained visual encoders
larger audio encoders
higher-resolution inputs
more sampled frames per video
larger batch sizes
faster repeated ablations
model-scaling experiments
full robustness evaluation
```

Important wording:

```text
The project does not depend entirely on A100 access, but A100 access would allow the model-scaling and foundation-model comparison to be carried out properly rather than being cut down due to hardware limits.
```

---

# Immediate To-Do List

## Today

* [ ] Create repo.
* [ ] Create project logbook.
* [ ] Create LaTeX paper folder.
* [ ] Load metadata CSV.
* [ ] Count A/B/C/D categories.
* [ ] Check file paths.
* [ ] Extract frames from 5 videos.
* [ ] Extract audio from 5 videos.
* [ ] Write dataset audit notes.
* [ ] Save 5–8 key papers.
* [ ] Draft one-page supervisor plan.

## Next 48 Hours

* [ ] Create cleaned manifest CSV.
* [ ] Build first identity-disjoint split.
* [ ] Create small balanced subset.
* [ ] Extract frames/audio from a tiny sample.
* [ ] Build visual dataset loader.
* [ ] Build audio dataset loader.
* [ ] Write research questions and hypotheses properly.
* [ ] Email supervisor with plan and meeting request.

## This Week

* [ ] Finish dataset audit.
* [ ] Finish mini literature review.
* [ ] Train tiny visual baseline.
* [ ] Train tiny audio baseline.
* [ ] Generate first sanity-check results.
* [ ] Confirm project scope with supervisor.
* [ ] Request GPU/storage access.

---

# One-Page Supervisor Plan Structure

```md
# MSc Project Scope

## Working Title
Modality-Specific Shortcut Learning in Audio-Visual Deepfake Detection

## Research Aim
Investigate whether multimodal deepfake detectors genuinely use both audio and visual evidence, or whether aggregate accuracy hides modality-specific failure modes.

## Dataset
FakeAVCeleb as main dataset.

## Key Issue
The dataset contains separate audio/video manipulation categories but is imbalanced, which may encourage modality bias.

## Proposed Experiments
1. Visual-only baseline
2. Audio-only baseline
3. Late fusion
4. Learned fusion
5. Natural vs balanced training
6. Missing/degraded modality robustness
7. Failure analysis
8. Optional model scaling with ViT/foundation encoders

## Main Contribution
A modality-aware diagnostic evaluation of multimodal deepfake detection, showing when fusion helps, when it fails, and whether models over-rely on one modality.

## Resource Request
16–24GB GPU minimum/preferred. A100-class GPU useful for larger encoders and model-scaling experiments.

## Questions
- Is this scope suitable?
- Is the contribution strong enough?
- Should I include external datasets?
- What should I prioritise first?
```

---

# Core Reminder

The project is not about proving that multimodal detection is amazing.

The project is about discovering:

```text
when multimodal detection helps,
when it fails,
why it fails,
and whether better training/fusion reduces those failures.
```
