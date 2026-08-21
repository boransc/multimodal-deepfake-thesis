# exp008_fakeavceleb_crossdataset results

## Dataset and validity

- Evaluation videos: 21,544
- Unique source identities represented: 500
- Visual decode coverage: 1.0000
- Audio decode coverage: 1.0000
- Both-modality decode coverage: 1.0000
- Ambiguous metadata method paths retained once: 22

## Pre-specified controlled-view results

- Visual-specific `visual_max` balanced accuracy: 0.7655
- Visual-specific `visual_max` AUC: 0.8346
- Audio-specific `audio_max` balanced accuracy: 0.5225
- Audio-specific `audio_max` AUC: 0.5770
- Binary `lr_fusion` balanced accuracy: 0.6002
- Binary `lr_fusion` AUC: 0.6358

Thresholds were selected only from AV-Deepfake1M++ validation scores. The learned fusion model was fitted only on AV-Deepfake1M++ training predictions. FakeAVCeleb labels were not used for training, threshold selection, pooling selection or fusion fitting.
