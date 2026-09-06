# Sep-TFAnet-VAD Grad-CAM

Grad-CAM interpretability for [Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD), a PyTorch
speaker-separation/VAD network — adapted from image-based Grad-CAM and validated on real LibriSpeech audio.

## Attribution

- **Base network and weights**: [MordehayM/Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD). This repo
  references and adapts that architecture (`network/model/model.py`, with one addition — see
  [Methodology](#methodology)) but does **not** redistribute the original repository's code or checkpoints.
- **Paper**:
  > Moradi, M., Gannot, S. et al. "Single-microphone speaker separation and voice activity detection in noisy and
  > reverberant environments." *EURASIP Journal on Audio, Speech, and Music Processing* (2025).
- **License note**: as of writing, the `Sep-TFAnet-VAD` repository does not publish a `LICENSE` file. No permissive
  reuse rights are assumed. This repo only references/attributes that network's architecture and weights for
  research/interpretability purposes; it does not redistribute them. You must obtain the weights directly from the
  original repository and are responsible for complying with whatever terms the original author sets.

## Setup

### Environment

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### Weights

Download `model_with_vad.pth` from [MordehayM/Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD) and place
it at:

```
weights/model_with_vad.pth
```

(`config_with_vad.json` under `configs/` is already included since it's a small text file describing the
architecture, not a weight.)

### LibriSpeech samples

No LibriSpeech audio is committed to this repo (it's copyrighted). Fetch two real utterances from two different
speakers with:

```bash
python scripts/fetch_librispeech_samples.py
```

This streams a couple of examples from `openslr/librispeech_asr` (Hugging Face `datasets`, streaming mode) so you
don't need to download the full ~346MB `test-clean.tar.gz`. It saves 2 utterances per speaker (4 files total) under
`data/librispeech_samples/`, since the sampler needs one utterance per speaker for the mixture plus a second as a
spare/anchor.

**Windows note**: Hugging Face's default audio decoder (`torchcodec`) requires FFmpeg shared libraries that may not
be present on your machine, causing a DLL load failure. The fetch script works around this by requesting
`Audio(decode=False)` and decoding the raw FLAC bytes manually with `soundfile`, so no FFmpeg install is required.

If Hugging Face streaming isn't reachable, the script falls back to `torchaudio.datasets.LIBRISPEECH(..., download=True)`,
which pulls the full `test-clean` subset. It never falls back to synthetic audio.

## Usage

```bash
# 1. Fetch 40 distinct real speakers (80 utterances) for 20 pairs
python scripts/fetch_librispeech_samples.py

# 2. Run target layer selection (scores all 24 TCN conv1d blocks)
python scripts/select_target_layer.py

# 3. Run single mixture verification figure
python scripts/verify_librispeech_gradcam.py --librispeech-root data/librispeech_samples --device cpu

# 4. Run multi-pair evaluation and statistical validation (20 pairs)
python scripts/evaluate_multi_pair_gradcam.py --librispeech-root data/librispeech_samples --device cpu
```

This pipeline will:
1. Stream 40 distinct real speakers from LibriSpeech `test-clean` to form 20 non-overlapping pairs.
2. Quantitatively score all 24 TCN conv1d candidate layers and select the optimal layer (`TCN.TCN.6.conv1d`).
3. Compute VAD-logit and waveform CAMs across all 20 pairs.
4. Output aggregate statistics (Mean ± Std), run a paired Wilcoxon signed-rank test against random controls, and save [multi_pair_results.csv](results/librispeech_gradcam/multi_pair_results.csv) and [paired_comparison_plot.png](results/librispeech_gradcam/paired_comparison_plot.png).

## Methodology

Three adaptation and evaluation decisions mattered and are worth calling out explicitly:

### 1. Pre-sigmoid VAD logit instead of the post-sigmoid probability

The VAD head ends in a `Sigmoid()`. Backpropagating from the post-sigmoid probability saturates almost everywhere
the model is confident (values pinned near 0 or 1), which collapses the upstream gradient and produces a
near-degenerate Grad-CAM. `network/model/model.py` exposes `model.vad_logits` — the same `VAD` module's
`output_layer_vad` output *before* the sigmoid (`VAD.forward_logits`, called from `SeparationModel.forward()`) — as a
non-saturating Grad-CAM target, without changing the model's normal sigmoid-probability output.

### 2. MAE instead of max-difference for comparing CAMs

An earlier iteration reported "two speakers' CAMs differ with max diff = 0.9999" as evidence of speaker-specific
attention. That number is not trustworthy on its own: max-elementwise-difference reads close to its theoretical
ceiling (~1.0) for *any* two sufficiently sparse activation maps — including one real CAM compared against a
**pure random-noise** map of the same shape — because sparse maps almost always have an isolated peak where the
other map is near zero. That's a property of sparsity, not evidence of disagreement.

This repo instead reports **mean absolute difference (MAE)**, plus the max-diff for context, for both an
independently min-max-normalized comparison and a shared-scale comparison, always alongside a random-noise control:

```
VAD-logit CAM, speaker vs speaker: MAE = 0.236 ± 0.096 (max diff ≈ 1.000)
VAD-logit CAM, speaker vs random noise: MAE = 0.418 ± 0.039 (max diff ≈ 0.995)
```

The max-diff numbers alone would suggest the real-vs-real and real-vs-random comparisons are equally "different."
The MAE numbers show the real speaker-vs-speaker CAMs are noticeably more self-similar to each other than either is
to random noise.

### 3. Principled layer selection via multi-metric scoring

Rather than picking an arbitrary layer index, all 24 TCN `conv1d` candidate layers (`TCN.TCN.0.conv1d` through `TCN.TCN.23.conv1d`) were systematically evaluated on a sample of LibriSpeech pairs across three quantitative metrics:
- **Gradient signal strength**: mean absolute gradient at that layer ($\text{mean}(|G|)$).
- **Representation richness**: variance of activations at that layer ($\text{var}(A)$).
- **CAM quality**: normalized Shannon entropy of the resulting Grad-CAM heatmap ($H_{\text{norm}}(\text{CAM}) \in [0, 1]$).

Each metric was z-scored across layers and averaged to produce a combined layer score.
- **Winning layer**: `TCN.TCN.6.conv1d` (Combined Score: **+0.7387**, Mean Gradient: $5.61 \times 10^{-4}$, Activation Variance: $3.48$, CAM Entropy: $0.852$).
- Top runners-up: `TCN.TCN.8.conv1d` (+0.5395) and `TCN.TCN.13.conv1d` (+0.4130).

Full layer rankings and bar charts are saved in [layer_scores.json](results/layer_selection/layer_scores.json) and [layer_scores.png](results/layer_selection/layer_scores.png).

## Results

Statistical validation across **$N = 20$ non-overlapping speaker pairs** (40 distinct speakers from LibriSpeech `test-clean`):

| Target Kind | Real Speaker-vs-Speaker MAE | Real-vs-Random Control MAE | Wilcoxon $W$ | $p$-value | Rank-Biserial $r$ |
|---|---|---|---|---|---|
| **VAD-Logit CAM** | **$0.2363 \pm 0.0959$** | $0.4177 \pm 0.0394$ | $1.0$ | **$3.81 \times 10^{-6}$** | **$0.990$** |
| **Waveform CAM** | **$0.2595 \pm 0.0772$** | $0.4146 \pm 0.0595$ | $6.0$ | **$2.67 \times 10^{-5}$** | **$0.943$** |

### Key Artifacts & Visualizations

- **Paired Comparison Plot**: [paired_comparison_plot.png](results/librispeech_gradcam/paired_comparison_plot.png) (shows real MAE vs random control MAE across all 20 speaker pairs).
- **Layer Selection Scores**: [layer_scores.png](results/layer_selection/layer_scores.png) (bar chart ranking all 24 TCN conv1d blocks).
- **Per-Pair CSV Data**: [multi_pair_results.csv](results/librispeech_gradcam/multi_pair_results.csv) (individual metrics for all 20 pairs).
- **Single Pair Example**: see `results/librispeech_gradcam/final/` (`example_speaker0.png`, `example_speaker1.png`, and `source_speakers_comparison.jpg`).

Real speaker-vs-speaker CAMs are statistically significantly more self-similar (lower MAE, $p < 0.0001$) than real-vs-random-noise across all 20 independent speaker pairs, with a very large effect size ($r > 0.94$).

## Limitations / Next Steps

- **Layer selection sample size**: The initial layer-selection scoring ablation was conducted on a 5-pair sample subset before running the 20-pair evaluation.
- **Corpus scope**: Validation was conducted on 2-speaker 3.0s mixtures from LibriSpeech `test-clean`. Performance on noisy, reverberant, or in-the-wild speech remains an avenue for future work.
- **Single winning layer evaluated**: Multi-pair validation was conducted using the optimal selected layer (`TCN.TCN.6.conv1d`).
- **CPU execution**: Benchmarks and verification were performed on CPU (`--device cpu`). GPU execution is supported via `--device cuda`.
