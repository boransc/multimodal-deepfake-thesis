# Visual Baseline Experiments — Experiment 002 and Experiment 003

This document records the two most recent **visual-only feature-cache baseline experiments** for the MSc deepfake detection project.

The experiments use **AV-Deepfake1M++ validation subset videos** and focus on visual-only detection using frozen ConvNeXt-Base features. They are intended to support the dissertation theme of **generalisation, modality-specific behaviour, and shortcut learning in audio-visual deepfake detection**.

---

## High-level purpose

The purpose of these two experiments was not simply to maximise accuracy. The aim was to establish whether a visual-only detector can distinguish real and fake videos under more controlled, group-disjoint evaluation settings, and whether it behaves differently across manipulation conditions:

- `real`
- `fake_video_fake_audio`
- `fake_video_real_audio`
- `real_video_fake_audio`

The especially important condition is:

```text
real_video_fake_audio
```

This condition should contain **real visual content** but **fake audio**. A visual-only model should not strongly classify these as fake based only on frames unless there are visual artefacts, encoding artefacts, dataset-level correlations, or other shortcut signals.

---

# Experiment overview

## Experiment 002

```text
Run name:
002_visual_convnext_base_frozen_features_linear_group_disjoint_balanced40k_16frames
```

Core setup:

```text
Frozen ConvNeXt-Base visual encoder
16 uniformly sampled frames per video
Mean-pooled frame embeddings
Linear classifier on cached visual features
Group-disjoint split
Balanced 40k target
```

Main role in project:

```text
Clean balanced visual-only baseline.
```

---

## Experiment 003

```text
Run name:
003_visual_convnext_base_frozen_features_linear_group_disjoint_balancedfull_val_16frames
```

Important correction:

```text
The folder name says 16frames, but the actual experiment used NUM_FRAMES = 64.
```

Actual setup:

```text
Frozen ConvNeXt-Base visual encoder
64 uniformly sampled frames per video
Mean-pooled frame embeddings
Linear classifier on cached visual features
Group-disjoint split
Larger semi-natural / imbalanced split
```

Main role in project:

```text
Larger visual-only stress test using more frames and more data.
```

---

# Shared methodology

Both experiments use the same overall feature-cache design.

## Why feature caching was used

The original raw-video training approach was too slow because every epoch re-decoded MP4 videos inside the DataLoader. A one-batch check for raw video training showed that loading a batch could be extremely slow, making full end-to-end training impractical.

The feature-cache approach avoids this by separating the expensive video-processing stage from the cheap classifier-training stage.

The pipeline becomes:

```text
video
→ sample frames once
→ frozen ConvNeXt-Base encoder
→ mean-pooled video feature vector
→ save cached feature shards to disk
→ train linear classifier on cached features
```

This means video decoding and ConvNeXt feature extraction happen once per experiment, rather than once per epoch.

---

## Visual feature extraction

For each video:

```text
1. Uniformly sample N frames.
2. Resize frames to 224 × 224.
3. Apply ImageNet normalisation.
4. Pass frames through frozen ConvNeXt-Base.
5. Remove/replace the final classification layer.
6. Obtain one feature vector per frame.
7. Mean-pool frame features to produce one video-level feature.
```

For ConvNeXt-Base, the resulting feature dimension is:

```text
1024
```

So each video becomes:

```text
[1024]-dimensional visual feature vector
```

---

## Classifier

Both experiments use a lightweight linear classifier on top of the frozen ConvNeXt features:

```text
LayerNorm(1024)
Linear(1024 → 2)
```

The classifier predicts:

```text
0 = real
1 = fake
```

This should be described as:

```text
Frozen ConvNeXt-Base visual encoder + linear classification head
```

It should not be described as an end-to-end fine-tuned ConvNeXt model.

---

## Split strategy

Both experiments use **group-disjoint splitting**. The group is based on the video clip folder, so related versions of the same underlying clip should not appear across train, validation, and test splits.

This is important because random row splitting can leak related clips across splits and inflate performance.

The group overlap checks were:

```text
train/val overlap  = 0
train/test overlap = 0
val/test overlap   = 0
```

for both experiments.

---

# Experiment 002 — Balanced 40k visual baseline

## Run identity

```text
Experiment ID:
002

Experiment name:
visual_convnext_base_frozen_features_linear_group_disjoint_balanced40k_16frames

Full run name:
002_visual_convnext_base_frozen_features_linear_group_disjoint_balanced40k_16frames
```

---

## Configuration

```text
Task:
visual-only binary classification using cached features

Dataset:
AV-Deepfake1M++ validation subset

Encoder:
Frozen ConvNeXt-Base

Classifier:
Linear classifier

Number of frames:
16

Image size:
224 × 224

Feature dimension:
1024

Feature batch size:
8

Number of DataLoader workers:
2

Prefetch factor:
1

Pin memory:
False

Classifier batch size:
2048

Classifier epochs:
30

Classifier learning rate:
1e-3

Classifier weight decay:
1e-4

Split type:
group-disjoint

Class balance:
balanced 40k target

Seed:
42
```

---

## Split summary

Experiment 002 used a balanced split:

| Split | Total | Real | Fake |
|---|---:|---:|---:|
| Train | 28,000 | 14,000 | 14,000 |
| Validation | 6,000 | 3,000 | 3,000 |
| Test | 6,000 | 3,000 | 3,000 |

Group overlap:

| Overlap check | Count |
|---|---:|
| Train/validation groups | 0 |
| Train/test groups | 0 |
| Validation/test groups | 0 |

---

## Condition distribution

### Train

| Condition | Count |
|---|---:|
| real | 14,000 |
| real_video_fake_audio | 4,721 |
| fake_video_real_audio | 4,701 |
| fake_video_fake_audio | 4,578 |

### Validation

| Condition | Count |
|---|---:|
| real | 3,000 |
| fake_video_fake_audio | 1,048 |
| fake_video_real_audio | 979 |
| real_video_fake_audio | 973 |

### Test

| Condition | Count |
|---|---:|
| real | 3,000 |
| real_video_fake_audio | 1,027 |
| fake_video_fake_audio | 1,001 |
| fake_video_real_audio | 972 |

---

## Feature extraction

A quick loader check produced:

```text
Frames: torch.Size([8, 16, 3, 224, 224])
Labels: torch.Size([8])
Indices: torch.Size([8])
Elapsed: 0.63s
```

Feature extraction output:

| Split | Feature tensor shape | Number of shards | Extraction time |
|---|---:|---:|---:|
| Train | `[28000, 1024]` | 3500 | 16.16 min |
| Validation | `[6000, 1024]` | 750 | 3.50 min |
| Test | `[6000, 1024]` | 750 | 3.55 min |

Total feature extraction time was approximately:

```text
23.21 minutes
```

This was much more practical than raw end-to-end video training.

---

## Final test metrics

| Metric | Value |
|---|---:|
| Loss | 0.5862 |
| Accuracy | 0.7018 |
| Balanced accuracy | 0.7018 |
| F1 | 0.7149 |
| Precision | 0.6849 |
| Recall | 0.7477 |
| AUC | 0.7674 |

Because the test set is balanced, raw accuracy and balanced accuracy are identical.

---

## Condition-level test results

| Condition | n | True label | Interpretation | Success rate | Predicted fake rate | Mean fake probability |
|---|---:|---:|---|---:|---:|---:|
| fake_video_fake_audio | 1001 | 1 | fake recall | 0.7532 | 0.7532 | 0.6121 |
| fake_video_real_audio | 972 | 1 | fake recall | 0.7490 | 0.7490 | 0.6112 |
| real | 3000 | 0 | real recall | 0.6560 | 0.3440 | 0.4339 |
| real_video_fake_audio | 1027 | 1 | fake recall | 0.7410 | 0.7410 | 0.6061 |

---

## Key interpretation

Experiment 002 gives a clean and credible visual-only baseline.

The model performs clearly above chance:

```text
Balanced accuracy: 70.18%
AUC:               76.74%
```

This indicates that frozen ConvNeXt-Base visual embeddings contain useful information for separating real and fake-labelled samples.

However, the per-condition results are more important than the headline score. The model predicts `real_video_fake_audio` as fake at a rate of:

```text
74.10%
```

This is notable because the model only sees visual frames, while this condition should have real visual content and fake audio.

Possible explanations include:

```text
1. Visual-side artefacts introduced during dataset construction.
2. Re-encoding/compression artefacts correlated with fake labels.
3. Dataset packaging or preprocessing differences between real and manipulated samples.
4. Source/video identity correlations that remain even under group-disjoint splitting.
5. A broader shortcut where fake-labelled files differ from real files in non-semantic visual ways.
```

This supports the project theme that deepfake detectors may rely on shortcuts rather than clean modality-specific forgery evidence.

---

## Strengths

```text
1. Balanced evaluation makes accuracy meaningful.
2. Group-disjoint split reduces clip-level leakage.
3. Feature caching makes the experiment computationally practical.
4. Train, validation, and test sizes are clean and easy to report.
5. Condition-level results directly support shortcut/modality analysis.
```

---

## Limitations

```text
1. ConvNeXt-Base was frozen, not fine-tuned end-to-end.
2. The classifier is linear, so the model capacity after feature extraction is limited.
3. Only 16 frames are sampled, so temporal information is weak.
4. The model has no access to audio, so it cannot truly verify audio manipulation.
5. The result cannot be directly compared to the earlier random-split experiment as a pure split-only comparison, because the modelling setup also changed.
```

---

## Suggested dissertation wording

> A frozen ConvNeXt-Base visual encoder with a linear classification head was evaluated on a group-disjoint balanced 40k split. Each video was represented by the mean-pooled embedding of 16 uniformly sampled frames. The model achieved 70.2% accuracy, 70.2% balanced accuracy, 71.5% F1 and 76.7% AUC. While this performance is clearly above chance, condition-level analysis showed that the visual-only model classified `real_video_fake_audio` samples as fake 74.1% of the time, despite these samples having real visual content. This suggests that the detector may be exploiting non-semantic visual or dataset-level artefacts correlated with fake labels rather than purely identifying visual manipulation.

---

# Experiment 003 — Larger 64-frame semi-natural visual baseline

## Run identity

```text
Experiment ID:
003

Experiment name:
visual_convnext_base_frozen_features_linear_group_disjoint_balancedfull_val_16frames

Full run name:
003_visual_convnext_base_frozen_features_linear_group_disjoint_balancedfull_val_16frames
```

Important correction:

```text
The run/folder name says 16frames, but the actual configuration used NUM_FRAMES = 64.
```

Correct label for reporting:

```text
Frozen ConvNeXt-Base, 64 frames, larger semi-natural imbalanced group-disjoint split
```

---

## Configuration

```text
Task:
visual-only full-val binary classification using cached features

Dataset:
AV-Deepfake1M++ validation subset

Encoder:
Frozen ConvNeXt-Base

Classifier:
Linear classifier

Actual number of frames:
64

Image size:
224 × 224

Feature dimension:
1024

Feature batch size:
32

Number of DataLoader workers:
2

Prefetch factor:
1

Pin memory:
False

Classifier batch size:
2048

Classifier epochs:
30

Classifier learning rate:
1e-3

Classifier weight decay:
1e-4

Split type:
group-disjoint

Class balance:
not actually balanced; larger semi-natural / imbalanced split

Seed:
42
```

---

## Naming issue

The experiment name contains:

```text
balancedfull_val_16frames
```

But the actual run used:

```text
NUM_FRAMES = 64
```

and the sanity check confirmed:

```text
Frames: torch.Size([32, 64, 3, 224, 224])
```

This should be explicitly corrected in any experiment table or write-up.

Recommended table name:

```text
003 — Frozen ConvNeXt-Base, 64 frames, larger imbalanced split
```

---

## Split summary

Experiment 003 used a larger imbalanced split:

| Split | Total | Real | Fake |
|---|---:|---:|---:|
| Train | 41,305 | 13,988 | 27,317 |
| Validation | 8,818 | 2,949 | 5,869 |
| Test | 8,953 | 3,084 | 5,869 |

Group overlap:

| Overlap check | Count |
|---|---:|
| Train/validation groups | 0 |
| Train/test groups | 0 |
| Validation/test groups | 0 |

---

## Important class imbalance note

This experiment is not balanced.

The test fake proportion is:

```text
5869 / 8953 = 65.55%
```

So an always-fake classifier would already achieve approximately:

```text
65.6% accuracy
```

Therefore, for this experiment, the most important metrics are:

```text
Balanced accuracy
AUC
Real recall
Condition-level recall
```

Raw accuracy and F1 should be interpreted carefully because they are affected by class imbalance.

---

## Condition distribution

### Train

| Condition | Count |
|---|---:|
| real | 13,988 |
| fake_video_real_audio | 9,140 |
| fake_video_fake_audio | 9,139 |
| real_video_fake_audio | 9,038 |

### Validation

| Condition | Count |
|---|---:|
| real | 2,949 |
| real_video_fake_audio | 2,027 |
| fake_video_fake_audio | 1,959 |
| fake_video_real_audio | 1,883 |

### Test

| Condition | Count |
|---|---:|
| real | 3,084 |
| fake_video_real_audio | 1,981 |
| real_video_fake_audio | 1,966 |
| fake_video_fake_audio | 1,922 |

---

## Feature extraction

A quick loader check produced:

```text
Frames: torch.Size([32, 64, 3, 224, 224])
Labels: torch.Size([32])
Indices: torch.Size([32])
Elapsed: 8.59s
```

Cached feature tensors:

| Split | Feature tensor shape |
|---|---:|
| Train | `[41305, 1024]` |
| Validation | `[8818, 1024]` |
| Test | `[8953, 1024]` |

For the test split:

```text
test: saved 8953 features across 280 shards
Elapsed: 12.14 min
```

Train and validation features already existed and were reused in the notebook run:

```text
train: existing complete shard manifest found, skipping extraction.
val: existing complete shard manifest found, skipping extraction.
```

---

## Final test metrics

| Metric | Value |
|---|---:|
| Loss | 0.5181 |
| Accuracy | 0.7551 |
| Balanced accuracy | 0.6832 |
| F1 | 0.8303 |
| Precision | 0.7605 |
| Recall | 0.9143 |
| AUC | 0.7971 |

---

## Condition-level test results

| Condition | n | True label | Interpretation | Success rate | Predicted fake rate | Mean fake probability |
|---|---:|---:|---|---:|---:|---:|
| fake_video_fake_audio | 1922 | 1 | fake recall | 0.9105 | 0.9105 | 0.7396 |
| fake_video_real_audio | 1981 | 1 | fake recall | 0.9117 | 0.9117 | 0.7410 |
| real | 3084 | 0 | real recall | 0.4520 | 0.5480 | 0.5283 |
| real_video_fake_audio | 1966 | 1 | fake recall | 0.9207 | 0.9207 | 0.7469 |

---

## Key interpretation

Experiment 003 produced a higher headline accuracy and AUC than Experiment 002:

```text
Accuracy: 75.51%
AUC:      79.71%
```

However, the model is clearly biased toward predicting fake.

The fake recall is very high:

```text
fake_video_fake_audio: 91.05%
fake_video_real_audio: 91.17%
real_video_fake_audio: 92.07%
```

but real recall is poor:

```text
real recall: 45.20%
```

This means the classifier correctly identifies many fake-labelled examples but falsely flags many real examples as fake.

The class imbalance helps explain this behaviour because the training, validation, and test sets are fake-heavy.

---

## Most important finding

The most important condition-level result is:

```text
real_video_fake_audio predicted fake rate: 92.07%
```

A visual-only model predicts this condition as fake extremely often despite the visual stream being labelled real.

This is even stronger than in Experiment 002, where `real_video_fake_audio` fake recall was:

```text
74.10%
```

In Experiment 003, with 64 frames and a larger imbalanced split, it becomes:

```text
92.07%
```

This suggests the model is not simply detecting obvious visual manipulation. It may be exploiting features correlated with fake-labelled videos, including possible encoding artefacts, dataset construction artefacts, or visual distribution differences.

---

## Fake bias

The model predicts fake very frequently.

For real videos:

```text
Predicted fake rate on real videos: 54.80%
Real recall:                       45.20%
```

This means more than half of genuinely real videos are incorrectly classified as fake.

The high F1 score should therefore not be over-interpreted, because F1 is calculated for the positive/fake class and the dataset is fake-heavy.

The better interpretation is:

```text
The model ranks examples moderately well by AUC, but the default 0.5 decision threshold and imbalanced training distribution lead to strong fake-class bias.
```

---

## Checkpoint selection issue

The classifier was saved based on best validation F1.

For an imbalanced fake-heavy dataset, selecting the checkpoint by validation F1 can favour a model that predicts the positive/fake class too often.

For this experiment type, better checkpoint criteria would be:

```text
validation balanced accuracy
```

or:

```text
validation AUC
```

This does not require rerunning feature extraction. If needed, only the linear classifier training needs to be rerun.

---

## Suggested threshold tuning

Because the test AUC is reasonably strong:

```text
AUC: 0.7971
```

the raw probability scores contain useful ranking information. The poor real recall may partly result from using the default decision threshold:

```text
fake if p(fake) >= 0.5
```

A useful follow-up is to tune the classification threshold on the validation set to maximise balanced accuracy, then apply that threshold once to the test set.

This could improve real recall without changing the feature extractor.

Recommended follow-up:

```text
1. Use validation probabilities.
2. Search thresholds from 0.05 to 0.95.
3. Select threshold with highest validation balanced accuracy.
4. Apply that threshold to test predictions.
5. Report both default-threshold and threshold-tuned metrics.
```

---

## Strengths

```text
1. Uses more videos than Experiment 002.
2. Uses 64 frames rather than 16.
3. Still uses group-disjoint splitting.
4. Feature extraction successfully scales to a larger setup.
5. AUC improves over Experiment 002.
6. Strongly exposes fake-class bias and possible visual shortcut learning.
```

---

## Limitations

```text
1. The run name is wrong because it says 16frames despite using 64 frames.
2. The split is not balanced, so accuracy and F1 are less reliable headline metrics.
3. The model is strongly biased toward fake predictions.
4. The classifier checkpoint was selected by validation F1, which is not ideal for imbalanced data.
5. The experiment is not directly comparable to Experiment 002 because both frame count and dataset distribution changed.
6. It is not a true full validation set experiment; it is a larger semi-natural subset.
```

---

## Suggested dissertation wording

> A second visual-only experiment used the same frozen ConvNeXt-Base feature extraction approach but increased the number of sampled frames to 64 and used a larger, imbalanced group-disjoint split. This model achieved 75.5% accuracy and 79.7% AUC, but balanced accuracy was lower at 68.3%. The condition-level results showed strong fake-class bias: fake-labelled conditions were detected with approximately 91–92% recall, while real recall was only 45.2%. Notably, `real_video_fake_audio` samples were classified as fake 92.1% of the time, despite the visual stream being labelled real. This further suggests that the visual-only detector may rely on dataset-level or encoding artefacts correlated with fake labels rather than purely visual manipulation evidence.

---

# Direct comparison

## Main setup comparison

| Item | Experiment 002 | Experiment 003 |
|---|---|---|
| Encoder | Frozen ConvNeXt-Base | Frozen ConvNeXt-Base |
| Classifier | Linear | Linear |
| Frames | 16 | 64 |
| Feature dim | 1024 | 1024 |
| Split | Group-disjoint | Group-disjoint |
| Dataset size | Balanced 40k target | Larger semi-natural / imbalanced |
| Train size | 28,000 | 41,305 |
| Val size | 6,000 | 8,818 |
| Test size | 6,000 | 8,953 |
| Real/fake balance | 50/50 | Fake-heavy |
| Best for | Clean balanced baseline | Larger imbalanced stress test |

---

## Main metric comparison

| Metric | Experiment 002 | Experiment 003 |
|---|---:|---:|
| Accuracy | 0.7018 | 0.7551 |
| Balanced accuracy | 0.7018 | 0.6832 |
| F1 | 0.7149 | 0.8303 |
| Precision | 0.6849 | 0.7605 |
| Recall | 0.7477 | 0.9143 |
| AUC | 0.7674 | 0.7971 |

---

## Condition comparison

| Condition | Experiment 002 success rate | Experiment 003 success rate |
|---|---:|---:|
| fake_video_fake_audio | 0.7532 | 0.9105 |
| fake_video_real_audio | 0.7490 | 0.9117 |
| real | 0.6560 | 0.4520 |
| real_video_fake_audio | 0.7410 | 0.9207 |

---

## What improved in Experiment 003?

Experiment 003 improved:

```text
1. Overall accuracy.
2. AUC.
3. Fake recall.
4. Detection rate for all fake-labelled conditions.
5. Scale of the visual feature extraction pipeline.
```

However, these improvements must be interpreted carefully because the dataset became fake-heavy.

---

## What got worse in Experiment 003?

Experiment 003 had worse:

```text
1. Balanced accuracy.
2. Real recall.
3. Class balance.
4. Interpretability of accuracy/F1 as headline metrics.
```

The real recall dropped from:

```text
Experiment 002: 65.60%
Experiment 003: 45.20%
```

This is a major issue and indicates strong fake bias.

---

# Overall interpretation across both experiments

Together, the two experiments show that frozen ConvNeXt-Base visual features are useful for deepfake detection, but the behaviour is not cleanly modality-specific.

The balanced 16-frame experiment showed moderate, credible performance:

```text
Balanced accuracy: 70.18%
AUC:               76.74%
```

The larger 64-frame imbalanced experiment improved AUC:

```text
AUC: 79.71%
```

but introduced strong fake-class bias:

```text
Real recall: 45.20%
```

The most consistent and project-relevant observation is that the visual-only model repeatedly classifies `real_video_fake_audio` samples as fake:

```text
Experiment 002: 74.10%
Experiment 003: 92.07%
```

Because these samples should have real visual content, this suggests that the model may be using shortcut signals correlated with fake labels rather than directly detecting visual manipulation.

---

# How to use these experiments in the dissertation

## Recommended table labels

Use these names in the dissertation rather than the raw folder names:

| Dissertation label | Actual run |
|---|---|
| Visual-16 Balanced | Experiment 002 |
| Visual-64 Imbalanced | Experiment 003 |

More descriptive labels:

```text
Visual-16 Balanced:
Frozen ConvNeXt-Base, 16 frames, balanced group-disjoint split

Visual-64 Imbalanced:
Frozen ConvNeXt-Base, 64 frames, larger imbalanced group-disjoint split
```

---

## Recommended results table

| Experiment | Frames | Split | Test size | Accuracy | Balanced accuracy | F1 | AUC |
|---|---:|---|---:|---:|---:|---:|---:|
| Visual-16 Balanced | 16 | Balanced group-disjoint | 6,000 | 0.7018 | 0.7018 | 0.7149 | 0.7674 |
| Visual-64 Imbalanced | 64 | Larger imbalanced group-disjoint | 8,953 | 0.7551 | 0.6832 | 0.8303 | 0.7971 |

---

## Recommended condition table

| Experiment | FVF-A fake recall | FVR-A fake recall | RVF-A fake recall | Real recall |
|---|---:|---:|---:|---:|
| Visual-16 Balanced | 0.7532 | 0.7490 | 0.7410 | 0.6560 |
| Visual-64 Imbalanced | 0.9105 | 0.9117 | 0.9207 | 0.4520 |

Where:

```text
FVF-A = fake_video_fake_audio
FVR-A = fake_video_real_audio
RVF-A = real_video_fake_audio
```

The abbreviation can be avoided in the final paper if clarity is more important than space.

---

# Main claims supported by these experiments

## Claim 1

```text
Frozen visual features provide above-chance deepfake detection on group-disjoint splits.
```

Supported by:

```text
Experiment 002 AUC: 0.7674
Experiment 003 AUC: 0.7971
```

---

## Claim 2

```text
The visual-only detector does not behave in a cleanly modality-specific way.
```

Supported by:

```text
real_video_fake_audio detected as fake:
Experiment 002: 74.10%
Experiment 003: 92.07%
```

---

## Claim 3

```text
Increasing frames/data can improve AUC but may worsen class bias under imbalanced distributions.
```

Supported by:

```text
AUC:
Experiment 002: 0.7674
Experiment 003: 0.7971

Real recall:
Experiment 002: 0.6560
Experiment 003: 0.4520
```

---

## Claim 4

```text
Balanced accuracy and condition-level metrics are necessary because headline accuracy/F1 can be misleading.
```

Supported by Experiment 003:

```text
Accuracy:          0.7551
F1:                0.8303
Balanced accuracy: 0.6832
Real recall:       0.4520
```

The high F1 is partly driven by high fake recall on a fake-heavy test set.

---

# Claims to avoid

Do not claim:

```text
Experiment 003 proves 64 frames are better than 16 frames.
```

Why:

```text
Frame count changed, but dataset size and class distribution also changed.
```

Do not claim:

```text
Experiment 002 and Experiment 003 are directly comparable controlled ablations.
```

Why:

```text
They differ in frame count, dataset size, and class balance.
```

Do not claim:

```text
The model definitely found visual manipulation in real_video_fake_audio.
```

Why:

```text
The visual stream should be real; the high fake prediction rate may indicate shortcuts or artefacts, not genuine visual forgery detection.
```

Do not claim:

```text
The model is end-to-end ConvNeXt.
```

Why:

```text
ConvNeXt is frozen and used as a feature extractor.
```

---

# Recommended next steps

## Step 1: Archive both experiments

Keep both experiments as visual baselines.

Do not rerun feature extraction unless necessary.

---

## Step 2: Run audio-only baseline on the same split

The next major experiment should be audio-only:

```text
004_audio_feature_cache_group_disjoint
```

Use the same or clearly documented split.

Recommended structure:

```text
audio waveform
→ pretrained audio feature encoder
→ cached audio features
→ linear or MLP classifier
→ condition-level evaluation
```

The key audio-only expectation:

```text
Audio-only model should perform strongly on:
real_video_fake_audio
fake_video_fake_audio

Audio-only model should struggle or behave differently on:
fake_video_real_audio
```

---

## Step 3: Run fusion baseline

After visual and audio features exist:

```text
visual feature vector + audio feature vector
→ fusion classifier
```

Fusion options:

```text
1. Score-level late fusion
2. Concatenated feature fusion + linear classifier
3. Concatenated feature fusion + small MLP
```

Start with score-level or concatenated linear fusion because it is simple and explainable.

---

## Step 4: Consider threshold tuning for Experiment 003

For Experiment 003, tune the decision threshold on validation data to maximise balanced accuracy.

This may improve real recall and make the imbalanced experiment more informative.

---

# Final summary

Experiment 002 is the cleaner baseline:

```text
Balanced split
16 frames
70.18% balanced accuracy
76.74% AUC
Moderate visual-only performance
Clear shortcut concern from real_video_fake_audio
```

Experiment 003 is the larger stress test:

```text
Larger imbalanced split
64 frames
75.51% accuracy
79.71% AUC
But only 68.32% balanced accuracy
Strong fake bias
Very low real recall
Even stronger real_video_fake_audio shortcut signal
```

Together, these experiments are useful because they show both:

```text
1. Visual-only features can detect fake-labelled samples above chance.
2. The detector likely exploits shortcuts or dataset-level artefacts rather than cleanly identifying visual manipulation alone.
```

This is directly aligned with the dissertation aim of investigating generalisation, robustness, and modality-specific limitations in multimodal deepfake detection.
