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
- 2k subjects and 1M deepfake videos using different manipulation methods.
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
- Makes same problem statement as previous paper, mentioning how it is an extension of the original.
- 3 Key Limitations of previous benchmarking:
Scale and sorce diversity were limited.
Benchmarks lacked generation diversity - less diversity encourage overfitting.
Streaming and redistribution artefacts uch as blur, recompression, frame drops, reverberation and packet jitters are overlooked despite being ubiquitous IRL. These can cause misleading signals, making the forgeries more complicated to detect.
- They solve this by increasing range of source datasets to VoxCeleb2, LRS3 and EngageNet. Deepfake generation pipeline includes 9 SOTA models to create unimodal & cross-modal forgeries using insert, replace and delete.
- Multiple techniques used to simulate real-world perturbations, including Gaussian/Poisson noise, rolling shutter, colour quantisation, Doppler shift, clipping, etc..
- FakeAVCeleb and LAV-DF are good multimodal deepfake datasets but both rely on a single visual and audio generator (Wav2Lip, SV2TTS), encouraging models to ovefit to method-specific artifacts and LAV-DF uses rule-based text manip. which limits diversity of generated deepfake content.
- Visual side, SOTA lip-sync models such as LatentSync, Diff2Lip and TalkLip outperform well known predecessors.
- Audio side, SOTA zero-shot TTS methods (XTTSv2, F5TTS) can clone a speaker's voice from seconds of reference audio while controllable prosody models can match emotion and style to perfect the deepfake.
- LLMs now deliver lower cost, more efficient and better quality outputs (GPT-4o mini mentioned) and can be used to automate planning insert/replace/delete operations and semantic editing while keeping meaning (already done in previous).
- They make a point that the community can push the performance for temporal localisation task as results from the challenge were low for this.


#### Why it matters to my project:
- AV-Deepfake1M++ is probably the strongest main dataset candidate because it directly supports audio-visual detection, temporal localisation, and robustness testing.
- It extends AV-Deepfake1M by increasing scale, source diversity, generation diversity, and perturbation realism.
- It is useful for my project because it tests whether models can handle partial manipulations, modality-specific manipulations, and real-world degradation.
- The dataset includes official benchmark splits and challenge-style evaluation, which could make experiments more reproducible.
- The 2025 challenge results suggest that video-level classification may already be strong for top teams, but temporal localisation still has a large performance gap.
- This supports a project contribution around temporal localisation, modality-aware fusion, perturbation robustness, or failure analysis.

#### Important facts:
- AV-Deepfake1M++ contains 2,051,154 videos, 1,423,218 fake videos, 627,936 real videos, around 502.8 million frames, 4,655.9 hours of video, and 7,109 subjects.
- It uses three real source datasets: VoxCeleb2, LRS3, and EngageNet.
- It creates content-driven manipulations using insert, replace, and delete operations.
- It uses a wider pool of audio and visual generation methods than LAV-DF and AV-Deepfake1M.
- It includes real, fake-audio-real-visual, real-audio-fake-visual, and fake-audio-fake-visual cases.
- It includes audio and visual perturbations to simulate real-world redistribution and streaming conditions.
- It provides frame-level and video-level annotations for classification and temporal localisation.
- The dataset is linked to the 2025 1M-Deepfakes Detection Challenge, with dataset and evaluation scripts available under a research-only license.



#### Limitations / problems / Future Directions:
- The dataset is extremely large, so full-scale training may be unrealistic for my MSc timeline without strong GPU and storage access.
- The paper’s own future directions suggest that perturbation-robust representation learning is still unsolved.
- Rapid adaptation to new forgery pipelines remains difficult because generators evolve quickly.
- Fine-grained multimodal reasoning is still needed, because detecting low-level artefacts may not be enough.
- Cross-cultural and multilingual robustness remains an open issue; MAVOS-DD may be relevant here, but it could become a separate project direction.
- Open-world evaluation is still a challenge because detectors need to generalise to unseen generators and unseen perturbations.
- Ethics, fairness, and privacy are major concerns because the dataset involves large-scale manipulated media.


#### How this connects to generalisation or robustness:
- Generalisation: the dataset splits use different identities, real sources, and generation methods across training/validation, TestA, and TestB, which helps test cross-domain generalisation.
- Robustness: the dataset includes perturbations such as blur, recompression, frame drops, noise, reverberation, packet jitter, audio stutter, and frame-rate jitter.
- Open-world evaluation: the paper argues that future detectors should handle unseen generation methods and unseen perturbations.
- Shortcut learning: older datasets with limited generators may encourage models to learn generator-specific artefacts rather than general deepfake cues.
- Modality robustness: because manipulations can affect audio, video, or both, the dataset can test whether multimodal models actually handle modality mismatch.


#### Corrections / additions after checking paper:
- They bridge the gap from previous works by sourcing mroe real data from multiple datasets, integrating more SOTA lip-sync and TTS methods, simulating 36 audio/visual IRL pertubation, and providing frame and video level annotations for both classification and temporal localisation.





## Themes Emerging
-

## Possible Research Gaps
-

## Useful References / BibTeX
-
