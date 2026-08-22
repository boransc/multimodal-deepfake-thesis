# Generalisation and Temporal Localisation in Audio-Visual Deepfake Detection

Supporting material for the MSc project **Generalisation and Temporal Localisation in Audio-Visual Deepfake Detection**.

This repository contains the experiment notebooks, helper code, preprocessing material, manifests, configurations, saved predictions and result files used in the project. Its purpose is to make the experimental process and the results reported in the dissertation inspectable.

## Important: dataset access and running the code

The repository is **not self-contained and cannot be run end-to-end as submitted** because the raw datasets used by the project are not included.

The project uses:

- **AV-Deepfake1M++**, using the usable validation subset for the source-domain experiments.
- **FakeAVCeleb**, used for the external zero-shot evaluation.

These datasets are large and are obtained separately from their official sources. They are not redistributed in this supporting material. The original experiments were also run in the QMUL project environment using local dataset paths and, for visual feature extraction, GPU resources.

As a result, an examiner can inspect the submitted source code, experiment configurations, split manifests, saved predictions, metrics and plots without access to the raw datasets, but **full media decoding, feature extraction and model reruns require separate access to the datasets**.

Large cached visual features and unnecessary large model checkpoints may also be omitted from the repository because of storage and repository-size constraints.

If the datasets are obtained separately, the main steps needed to rerun the project are described in **Rerunning the experiments** below.

## Repository structure

```text
.
├── data/
│   ├── avdeepfake1mpp/
│   └── fakeavceleb/
├── experiments/
│   ├── exp001_visual_convnext_random/
│   ├── exp002_visual_convnext_group_disjoint/
│   ├── exp003_visual_convnext_full_val/
│   ├── exp004_visual_temporal_bal40k/
│   ├── exp005_audio_temporal_bal40k/
│   ├── exp006_visual_temporal_fullval/
│   ├── exp007_late_fusion_bal40k/
│   └── exp008_fakeavceleb_crossdataset_eval/
├── src/
│   └── data_preparation/
│       ├── audio_audit.ipynb
│       └── manifest.ipynb
├── README.md
└── requirements.txt
```

The exact contents of each experiment folder vary because the experiments were developed at different stages of the project. In general, they contain the main notebook together with the relevant configuration, manifests, saved predictions, metrics, plots and helper code.

## Data preparation

### AV-Deepfake1M++

Only the **validation subset** of AV-Deepfake1M++ was used for the source-domain experiments.

The preprocessing/audit stage produced a usable working manifest of **76,928 videos** after filtering files that did not meet the required media checks.

Relevant preprocessing notebooks are:

```text
src/data_preparation/audio_audit.ipynb
src/data_preparation/manifest.ipynb
```

The repository contains the derived manifests required to document the final experiment setup. Raw AV-Deepfake1M++ videos are not included.

### FakeAVCeleb

FakeAVCeleb was used only for the external evaluation in Exp008.

The external evaluation uses audited metadata and source-matched subsets constructed from the locally available FakeAVCeleb dataset. Raw FakeAVCeleb videos are not included.

## Experiment progression

| Experiment | Purpose / main change |
| --- | --- |
| **Exp001** | Initial random-split visual baseline. This was later treated as a diagnostic experiment after the split was found to allow overly similar source content across partitions. |
| **Exp002** | Balanced clip-group-disjoint visual baseline. |
| **Exp003** | Larger naturally fake-heavy visual clip experiment using 64 sampled frames. |
| **Exp004** | Visual temporal localisation over 64 sampled positions. |
| **Exp005** | Corrected audio temporal experiment using decoded 16 kHz audio and log-STFT features. |
| **Exp006** | Visual temporal localisation on the larger natural source-domain distribution. |
| **Exp007** | Temporal pooling and rule-based / learned late fusion. |
| **Exp008** | Zero-shot transfer to FakeAVCeleb with source-matched evaluation and dataset auditing. |

The detailed results are stored inside the experiment folders and reported in the dissertation paper.

## Environment

The project was developed in a Jupyter-based Python environment.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

**FFmpeg/ffprobe are also required** for media probing and audio extraction and must be installed separately as system dependencies.

A CUDA-capable GPU is strongly recommended for ConvNeXt feature extraction. The project used university GPU resources for the computationally expensive stages.

## Rerunning the experiments

The original notebooks were developed with project root:

```text
/home/jovyan/MSC_PROJECT
```

The repository therefore should not be expected to run immediately on a different machine without first configuring the required data and paths.

To rerun the full pipeline:

1. Obtain **AV-Deepfake1M++** and/or **FakeAVCeleb** separately through their official access routes.
2. Place the datasets in local directories accessible to the notebooks.
3. Update `PROJECT_ROOT` and dataset-path variables near the beginning of the relevant notebooks.
4. Install the Python dependencies from `requirements.txt`.
5. Install FFmpeg/ffprobe and ensure they are available on the system `PATH`.
6. Run the AV-Deepfake1M++ audit/manifest preparation notebooks if rebuilding the source manifest.
7. Run the required experiments in numerical order where later experiments depend on earlier outputs.

Relevant dependencies between experiments are:

- Exp004 and Exp005 reuse the controlled Exp002 split.
- Exp007 uses the saved visual/audio temporal outputs from Exp004 and Exp005.
- Exp008 uses fixed source-domain components and the locally available FakeAVCeleb data for external evaluation.

Because the datasets and large cached features are not included, **the complete raw-video-to-result pipeline cannot be reproduced from this repository alone**. The submitted saved outputs are included so that the final experiment results and analysis can still be inspected without rerunning the full datasets.

## Reproducibility notes

- **Exp001** is an early diagnostic experiment and is not used as the main controlled baseline.
- **Exp002 and Exp003** contain cleaned reconstructions of the final experiment setup based on the archived configurations, manifests and saved results. This is stated inside the notebooks.
- **Exp005** is the corrected valid audio run. An earlier failed audio-decoding run was rejected and is not used as an empirical result.
- **Exp007** uses an exploratory learned stacker fitted from in-sample source-domain base-model training predictions rather than out-of-fold predictions.
- **Exp008** is zero-shot: FakeAVCeleb labels were not used to retrain the source models, tune source-domain thresholds or refit the learned fusion model.
- Raw datasets, large feature caches and unnecessary large checkpoints are not included.

## Inspecting the submitted results without dataset access

The project does not require the examiner to rerun the full datasets in order to inspect the evidence used in the dissertation.

The experiment directories contain saved outputs such as:

- configurations;
- split manifests;
- prediction CSV files;
- evaluation metrics;
- condition-level summaries;
- plots;
- notebook outputs.

These files provide the trace from the submitted code and experiment configuration to the numerical results discussed in the dissertation.

## Main software

The implementation primarily uses PyTorch/torchvision, NumPy, pandas, scikit-learn, OpenCV and Matplotlib, with FFmpeg used for media probing and audio extraction.
