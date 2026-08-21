# exp007_late_fusion_bal40k Results Summary

## Sources

- Visual source: `exp004_visual_temporal_bal40k`
- Audio source: `exp005_audio_temporal_bal40k`

## Task

Late fusion for binary any-fake detection on the balanced group-disjoint split.

## Best test method by balanced accuracy

- Method: `lr_fusion_all_pooling`
- Balanced accuracy: `0.6193`
- Accuracy: `0.6193`
- F1: `0.6026`
- Precision: `0.6303`
- Recall: `0.5773`
- AUC: `0.6639`

## Report points

1. Compare visual-only, audio-only, and fusion on the same common test subset.
2. Prioritise balanced accuracy and AUC over raw accuracy.
3. Use condition metrics to show whether each modality contributes correctly.
4. Discuss whether OR fusion improves fake recall but increases false positives.
5. Discuss whether learned fusion weights the modalities better than simple OR fusion.
