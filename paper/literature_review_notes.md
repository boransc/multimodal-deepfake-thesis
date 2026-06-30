# Literature Review Notes

## Papers to Read

| Citation Key                         | Paper                                                                                                                      |        Year | Topic                                      | Why I'm Reading It                                                                                                                                                                                                                                                                   | Status   |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ----------: | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| `wang_2021_m2tr`                     | M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection                                                          |        2021 | Multimodal Transformer                     | Understand an early transformer-based multimodal deepfake detection approach.                                                                                                                                                                                                        | Not Read |
| `tan_2021_efficientnetv2`            | EfficientNetV2: Smaller Models and Faster Training                                                                         |        2021 | Backbone / CNN                             | Understand efficient CNN backbones that could be used for visual baselines.                                                                                                                                                                                                          | Not Read |
| `tan_2019_efficientnet`              | EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks                                                   |        2019 | Backbone / CNN                             | Understand EfficientNet as a common scalable visual feature extractor.                                                                                                                                                                                                               | Not Read |
| `zhao_2025_deepfakebenchmm`          | DeepfakeBench-MM: A Comprehensive Benchmark for Multimodal Deepfake Detection                                              |        2025 | Benchmark                                  | Understand recent benchmarking practice for multimodal deepfake detection.                                                                                                                                                                                                           | Not Read |
| `gandhi_2024_a`                      | A Multimodal Framework for Deepfake Detection                                                                              |        2024 | Multimodal Detection                       | Study a general multimodal framework and compare its design choices with other approaches.                                                                                                                                                                                           | Not Read |
| `chollet_2016_xception`              | Xception: Deep Learning with Depthwise Separable Convolutions                                                              |        2016 | Backbone / CNN                             | Understand Xception, a classic backbone used in visual deepfake detection.                                                                                                                                                                                                           | Not Read |
| `katamnenivinayasree_2023_misavoidd` | MIS-AVoiDD: Modality Invariant and Specific Representation for Audio-Visual Deepfake Detection                             |        2023 | Representation Learning                    | Study modality-invariant and modality-specific representations, which is highly relevant to generalisation.                                                                                                                                                                          | Not Read |
| `rssler_2019_faceforensics`          | FaceForensics++: Learning to Detect Manipulated Facial Images                                                              |        2019 | Dataset / Visual Benchmark                 | Understand one of the main visual deepfake datasets and standard evaluation setups.                                                                                                                                                                                                  | Not Read |
| `alfasly_2022_learnable`             | Learnable Irrelevant Modality Dropout for Multimodal Action Recognition on Modality-Specific Annotated Videos              |        2022 | Multimodal Robustness                      | Explore modality dropout ideas that may transfer to robustness in audio-visual deepfake detection.                                                                                                                                                                                   | Not Read |
| `hashmi_2025_avtenet`                | AVTENet: A Human-Cognition-Inspired Audio-Visual Transformer-Based Ensemble Network for Video Deepfake Detection           |        2025 | Multimodal Transformer                     | Study a recent audio-visual transformer/ensemble method for deepfake detection.                                                                                                                                                                                                      | Not Read |
| `geirhos_2020_shortcut`              | Shortcut Learning in Deep Neural Networks                                                                                  |        2020 | Generalisation / Dataset Bias              | Understand shortcut learning, dataset bias and why models may fail under distribution shift.                                                                                                                                                                                         | Not Read |
| `oorloff_2024_avff`                  | AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection                                                             |        2024 | Feature Fusion                             | Study feature-level audio-visual fusion for multimodal detection.                                                                                                                                                                                                                    | Not Read |
| `li_2019_celebdf`                    | Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics                                                         |        2019 | Dataset / Visual Benchmark                 | Understand Celeb-DF as a challenging visual dataset for cross-dataset evaluation.                                                                                                                                                                                                    | Not Read |
| `jiang_2020_deeperforensics10`       | DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection                                           |        2020 | Dataset / Robustness                       | Understand a dataset designed for more realistic face forgery detection conditions.                                                                                                                                                                                                  | Not Read |
| `yamagishi_2021_asvspoof`            | ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection                                              |        2021 | Audio Benchmark                            | Understand modern audio spoofing/deepfake detection benchmarks.                                                                                                                                                                                                                      | Not Read |
| `oquab_2023_dinov2`                  | DINOv2: Learning Robust Visual Features without Supervision                                                                |        2023 | Visual Representation Learning             | Investigate whether robust self-supervised visual features could be useful for generalisation.                                                                                                                                                                                       | Not Read |
| `tak_2022_automatic`                 | Automatic Speaker Verification Spoofing and Deepfake Detection using wav2vec 2.0 and Data Augmentation                     |        2022 | Audio Detection                            | Learn how wav2vec 2.0 and augmentation are used for audio deepfake/spoofing detection.                                                                                                                                                                                               | Not Read |
| `wu_2024_deep`                       | Deep Multimodal Learning with Missing Modality: A Survey                                                                   |        2024 | Multimodal Learning Survey                 | Understand multimodal learning problems such as missing modalities, modality imbalance and robustness.                                                                                                                                                                               | Not Read |
| `todisco_2019_asvspoof`              | ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection                                                         |        2019 | Audio Benchmark                            | Learn the earlier ASVspoof benchmark setup and how audio spoofing evaluation evolved.                                                                                                                                                                                                | Not Read |
| `yang_2023_avoiddf`                  | AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake                                                               |        2023 | Joint Audio-Visual Learning                | Study joint learning between audio and visual modalities for deepfake detection.                                                                                                                                                                                                     | Not Read |
| `jeswani_2026_exploring`             | Exploring Facial Feature Extraction and Temporal Dynamics for Deepfake Video Detection: A CNN-RNN Study on FaceForensics++ |        2026 | Visual / Temporal Detection                | Understand facial feature extraction and temporal modelling for visual deepfake detection.                                                                                                                                                                                           | Not Read |
| `khalid_2022_fakeavceleb`            | FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset                                                               |        2022 | Dataset / Audio-Visual Benchmark           | Assess FakeAVCeleb as the likely main paired audio-video dataset for the project.                                                                                                                                                                                                    | Read     |
| `almutairi_2022_a`                   | A Review of Modern Audio Deepfake Detection Methods: Challenges and Future Directions                                      |        2022 | Audio Survey                               | Build background knowledge on audio deepfake detection methods, datasets and limitations.                                                                                                                                                                                            | Not Read |
| `zhang_2025_audio`                   | Audio Deepfake Detection: What Has Been Achieved and What Lies Ahead                                                       |        2025 | Audio Survey                               | Understand recent progress, remaining challenges and research gaps in audio deepfake detection.                                                                                                                                                                                      | Not Read |
| `salvi_2023_a`                       | A Robust Approach to Multimodal Deepfake Detection                                                                         |        2023 | Multimodal Detection / Robustness          | Study a robustness-focused multimodal deepfake detection method.                                                                                                                                                                                                                     | Not Read |
| `snehamuppalla_2023_integrating`     | Integrating Audio-Visual Features For Multimodal Deepfake Detection                                                        |        2023 | Feature Fusion                             | Compare another approach to integrating audio and visual information.                                                                                                                                                                                                                | Not Read |
| `khalid_2021_evaluation`             | Evaluation of an Audio-Video Multimodal Deepfake Dataset using Unimodal and Multimodal Detectors                           |        2021 | Unimodal vs Multimodal Evaluation          | Understand how unimodal and multimodal detectors are compared on audio-video deepfake data.                                                                                                                                                                                          | Not Read |
| `_2024_visual`                       | Visual Deepfake Detection: Review of Techniques, Tools, Limitations, and Future Prospects                                  |        2024 | Visual Survey                              | Build background knowledge on visual deepfake detection techniques and limitations.                                                                                                                                                                                                  | Not Read |
| `alrashoud_2025_deepfake`            | Deepfake Video Detection Methods, Approaches, and Challenges                                                               |        2025 | Video Survey                               | Understand current video deepfake detection approaches and open challenges.                                                                                                                                                                                                          | Not Read |
| `ilyas_2023_avfakenet`               | AVFakeNet: A Unified End-to-End Dense Swin Transformer Deep Learning Model for Audio-Visual Deepfakes Detection            |        2023 | Multimodal Swin Transformer                | Study an end-to-end audio-visual deepfake detector using transformer-style architecture.                                                                                                                                                                                             | Not Read |
| `cai_2023_avdeepfake1m`              | AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset                                                      |        2023 | Dataset / Audio-Visual Benchmark           | Understand a newer large-scale audio-visual deepfake dataset beyond FakeAVCeleb.                                                                                                                                                                                                     | Read     |
| `cai_2025_avdeepfake1m`              | AV-Deepfake1M++: A Large-Scale Audio-Visual Deepfake Benchmark with Real-World Perturbations                               |        2025 | Dataset / Robustness Benchmark             | Study robustness evaluation using real-world perturbations, highly relevant to your project direction.                                                                                                                                                                               | Not Read |
| `chandra_2024_deepfakeeval2024`      | Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024                                    | 2024 / 2025 | Dataset / In-the-wild multimodal benchmark | Useful as an external real-world generalisation test. Better for checking whether detectors trained on academic datasets survive real-world social-media/user-uploaded deepfakes. Probably not ideal as the main training/localisation dataset unless temporal labels are available. | Not Read |
| `croitoru_2025_mavosdd`              | MAVOS-DD: Multilingual Audio-Video Open-Set Deepfake Detection Benchmark                                                   |        2025 | Dataset / Multilingual open-set benchmark  | Useful for testing generalisation across unseen languages and unseen generation models. Strong candidate if the project shifts toward multilingual/open-set audio-visual deepfake detection rather than temporal localisation.                                                       | Not Read |

## Paper Notes

### FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset

Citation key: khalid_2022_fakeavceleb
Pages read: Full paper

#### What I think the paper is about:

- Deepfakes are a security and privacy issue because it can be used to impersonate a person in videos/images/audio.
- Their proposal is to create FakeAVCeleb that has video + audio and used popular deepfake methods.
- DNNs are main cause for rise in deepfakes.
- Most common deep learning-based generation methods use Autoencoders (AEs), Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs) to combine/superimpose a source human face image onto a target image.
- Deep learning methods are also used for voice cloning, but the paper describes voice cloning separately as a network-based speech synthesis problem. Examples include deepfakes former presidents with highly accurate lip-sync.
- High-quality deepfake datasets are needed to train and evaluate effective deepfake detection methods.
- While other deepfake datasets have been proposed to help with this, most only focus on realistic deepfake videos but don't consider audio and vice versa, being a limitation of many deepfake datasets.
- Conducted experiments using 11 different unimodal, ensemble-based and multimodal baseline deepfake methods (can check references to see the methods in depth) and compared with other datasets.
- FF++ and DFDC are very large datasets and widely used, one having video only and other with video and audio respectively, but DFDC label whole video as fake, not specifying
- FakeAVCeleb addresses these issues and has detailed labelling as well as videos w/ audio.
- Used Faceswap and Faceswap GAN (FSGAN) to generated swapped deepfake videos, and used a transfer learning-based real-time voice cloning tool (SV2TTS) to generate cloned audios then used Wav2Lip to reenact videos based on fake audio.
- Used Face++ to measure similarity of 2 faces to find most similar source/target pairs for more realistic deepfakes.
- Filtered generated videos based on certain criterias to ensure good quality dataset.
- Compared with other datasets using Capsule, HeadPose, Visual Artifacts (VA)-MLP/LogReg, Xception, Meso4, and MesoInception
- Summaries of results show EfficientNet-B0 maintained the best performances across Unimodal and Ensemble.

#### Why it matters to my project:

- Taught me about deepfake generation and detection methods, and will likely be a dataset used in the project
- Has Real-Real, Real-Fake, Fake-Real and Fake-Fake videos+audio
- Helps justify project focus on limitaions of multimodal detection methods
- Multimodal detection is not automatically better, especially when only one modality is manipulated.

#### Limitations / problems:

- Have not made use of deepfake polishing methods to remove artifacts caused by deepfake generation methods
- Dataset size is smaller compared to other large-scale datasets that are widely used (FF++, DFDC)
- In Unimodal Results, for baseline trained on audio only, Meso4 overgit real and fake class for audio detection, and the tested visual deepfake detection baselines were not well suited to audio-only detection; the authors suggest that speech verification or spoofing-detection models may be more appropriate.
- Multimodal results are poor, possibly because they were designed to perform specific tasks, and multimodal methods make it challenging to detect deepfakes when either the video or audio is fake. Therefore more research is needed in multimodal deepfake detectors.

#### How this connects to generalisation or robustness:

- Possible shortcut learning issue: if generated deepfakes contain artefacts from specific tools such as Faceswap, SV2TTS etc, models may learn those tool-specific artefacts rather than general deepfake features.
- This connects more strongly to generalisation than robustness. For robustness, I would need extra tests such as compression, noise, degraded audio, blurred frames, or missing modalities.

#### What I still don't understand:

- Deep learning-based generation methods (AEs, VAEs, GANs)
- FS, FSGAN, SV2TTS, Wav2Lip
- Deepfake detection methods
- Deepfake polishing methods

#### Other notes:

- Can check paper for definitions of dataset generated methods and deepfake detection methods used.

#### Corrections / additions after checking paper:

- FakeAVCeleb contains 500 real videos and 19,500 fake videos, for 20,000 total videos.
- The four categories are ARVR (500), AFVR (500), ARVF (9,000), and AFVF (10,000).
- AFVR is not lip-synced, while ARVF and AFVF are lip-synced.
- Main benchmark metric is frame-level AUC.
- EfficientNet-B0 is strongest/stablest across video-only and ensemble settings, but VGG achieves the best audio-only AUC.
- The paper includes limited compressed-video evaluation, but not a full robustness study.
- Real videos were selected from VoxCeleb2: 500 videos, one per celebrity, with an average duration of 7.8 seconds. The authors selected clear, centred, English-speaking face videos with minimal occlusion.

### AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset

Citation key: cai_2023_avdeepfake1m
Pages read: Full paper

#### What I think the paper is about:

- Deepfake dataset geenrated using an LLM
- > 2k subjects and 1M deepfake videos using different manipulation methods.
- Most datasets assume entirity of content is either real or fake, they don't consider the manipulation of small segments in the otherwise real content, making the content a deepfake in that regard. This can cause for the underlying meaning of the original content to be changed/manipulated, possibly refelcting a different message than intended originally.
- Their contribtion is the AV dataset used for classification and temporal localisation tasks.
- LAV-DF introduced first content-driven deepfake dataset for temporal localisation but was limited in quality and scale, issues they address.
- LAV-DF uses rule-based system to find antonyms to change sentiment in transcription manipulation. They argue this causes context inconsistencies, which they solve using LLM, resulting in diverse and context-consistent deepfake content.
- They also argue Wav2Lip and SV2TTS output quality is inadequate, instead using open-source SOTA methods for high-quality audio and video generation (does the same argument apply for FakeAVCeleb? They also used Wav2Lip and SV2TTS).
- Used subset of real videos from Voxceleb2 (same as FakeAVCeleb).
- Comparison of audio quality shows AV-Deepfake1M with higher audio quality compared to other datasets (FakeAVCeleb, LAV-DF).
- They conducted a user study with 25 participants to compare difficulty of spotting a deepfake between AV-Deepfake1M and LAV-DF. Conclusion is AV-Deepfake1M is more challenging and difficult for humans.
- Temporal deepfake localisation is audio, deepfake detection is video. They benchmarked these two using multiple SOTA methods for the respective deepfake.
- Used average precision (AP) and average recall (AR) for temporal deepfake localisation
- Used video-level accuracy (Acc.) and area under the curve (AUC) for deepfake detection
- Results indicate existing temporal deepfake localisation methods are falling behind, creating gap for further research (CONSIDER)
- Models with only video-level access and zero-shot performed poorly, except models designed to be generalisable. Even after providing further level/label access, the AUC of best perofrmers is less than 70.
- Expectedly visual-only methods performed consistently better on subset V (video-only modification i think) than the fullset, same for audio-only with subset A.
- Comparison of performance on temporal localisation and classification on AV-Deepfake1M and LAV-DF shows significant drops in performance on AV-Deepfake1M than LAV-DF, indicating that their dataset is more challenging.
- Pretrained Sception and BA-TFD on AV-Deepfake1M the finetuned and evaluated on LAV-DF show better results than models trained from scratch on LAV-DF.

#### Why it matters to my project:

- Gives the idea of possibly pursuing temporal deepfake localisation methods, as they seem to be falling behind based on the results of the paper.
- It gives me a sharper project direction: not just “is the whole video fake?”, but “where is the fake part, and is it audio, video, or both?”
- It connects to generalisation because methods that perform well on older datasets drop in performance on AV-Deepfake1M.
- It connects to robustness because partial, realistic, modality-specific manipulations are harder than simple whole-video labels.
- It suggests that audio-visual deepfake detection needs to reason about both modality and time.
- It may be a better direction than generic multimodal classification, because localisation gives a clearer research problem.

#### Limitations / problems:

- Similarly to other deepfake datasets, they also have a misbalance of fake and real videos.

#### How this connects to generalisation or robustness:

- Generalisation: test whether models trained on one dataset or manipulation style still work on another dataset or newer manipulation style.
- Robustness: test whether models still work when the video/audio is degraded, compressed, noisy, blurred, or partially manipulated.
- AV-Deepfake1M is more about generalisation and temporal localisation than robustness.

#### What I still don't understand:

- What does "in-the-wild" mean when referring to FF++ and DFDC?

#### Other notes:

- Check Figure 2 for Data manipulation and generation pipeline.
- It seems like AV-Deepfake1M and FakeAVCeleb have different strategies of making their deepfakes, one manipulates segments of the original content to make it a deepfake while the other seems to changes the whole content entirely?

#### Corrections / additions after checking paper:

- The LLM is used mainly for transcript manipulation, not for generating the whole dataset by itself.
- The dataset is about partial deepfakes: small audio/video segments are manipulated inside otherwise real videos.
- Temporal deepfake localisation means finding where the fake segment starts and ends.
- Deepfake detection usually means classifying the whole video as real or fake.
- Subset V means the visual modality is relevant because audio-only modified videos are excluded.
- Subset A means the audio modality is relevant because visual-only modified videos are excluded.
- AV-Deepfake1M argues that Wav2Lip and SV2TTS used in older datasets produce lower-quality outputs than the newer methods used here.
- AV-Deepfake1M contains 2,068 subjects, 286,721 real videos, 860,039 fake videos, and 1,146,760 total videos.
- It contains around 1,886 hours of audio-visual data.
- The dataset supports both temporal localisation and binary classification.
- It uses three text manipulation operations: replacement, deletion, and insertion.
- It uses three modality conditions: fake audio + fake visual, fake audio + real visual, and real audio + fake visual.
- The paper compares AV-Deepfake1M with LAV-DF and shows that existing methods perform worse on AV-Deepfake1M.
- The paper also includes a human study showing that humans find AV-Deepfake1M harder than LAV-DF.
- The dataset generation required around 3,000 GPU hours, but that was for creating the dataset, not simply training a detector.

### AV-Deepfake1M++: A Large-Scale Audio-Visual Deepfake Benchmark with Real-World Perturbations

Citation key: cai_2025_avdeepfake1m
Pages read:

#### What I think the paper is about:

-

#### Why it matters to my project:

-

#### Limitations / problems:

-

#### How this connects to generalisation or robustness:

-

#### What I still don't understand:

-

#### Other notes:

-

#### Corrections / additions after checking paper:

-

## Themes Emerging

-

## Possible Research Gaps

-

## Useful References / BibTeX

-
