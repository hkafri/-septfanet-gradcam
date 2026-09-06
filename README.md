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

# 4. Run multi-pair evaluation and statistical validation (Selection Set: 20 pairs)
python scripts/evaluate_multi_pair_gradcam.py --librispeech-root data/librispeech_samples --num-pairs 20 --seed-offset 1000

# 5. Fetch disjoint held-out set (30 new speakers / 15 pairs from validation split)
python scripts/fetch_librispeech_holdout.py

# 6. Run selection-bias-free held-out evaluation (15 pairs, target layer TCN.TCN.9.conv1d fixed)
python scripts/evaluate_multi_pair_gradcam.py --librispeech-root data/librispeech_holdout --num-pairs 15 --seed-offset 2000 --csv-output results/librispeech_gradcam/holdout_results.csv --summary-output results/librispeech_gradcam/holdout_summary.json --plot-output results/librispeech_gradcam/holdout_paired_comparison_plot.png
```

This pipeline will:
1. Stream 40 distinct real speakers from LibriSpeech `test-clean` to form 20 non-overlapping selection/evaluation pairs.
2. Quantitatively score all 24 TCN conv1d candidate layers using a non-monotonic entropy penalty and select the optimal layer (`TCN.TCN.9.conv1d`).
3. Fetch a completely disjoint set of 30 new speakers (15 pairs) from LibriSpeech `validation` (dev-clean) with zero speaker overlap.
4. Evaluate Grad-CAM across both sets and report selection-set, selection-bias-free held-out, and pooled aggregate statistics ($N = 35$ total pairs).

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
VAD-logit CAM, speaker vs speaker: MAE = 0.256 ± 0.109 (max diff ≈ 1.000)
VAD-logit CAM, speaker vs random noise: MAE = 0.409 ± 0.057 (max diff ≈ 0.934)
```

The max-diff numbers alone would suggest the real-vs-real and real-vs-random comparisons are equally "different."
The MAE numbers show the real speaker-vs-speaker CAMs are noticeably more self-similar to each other than either is
to random noise.

### 3. Principled layer selection via non-monotonic entropy scoring

Rather than picking an arbitrary layer index or monotonically rewarding diffuse heatmaps, all 24 TCN `conv1d` candidate layers (`TCN.TCN.0.conv1d` through `TCN.TCN.23.conv1d`) were systematically evaluated across all $N=20$ LibriSpeech speaker pairs using three quantitative metrics:
- **Gradient signal strength**: mean absolute gradient at that layer ($\text{mean}(|G|)$, log-transformed & z-scored).
- **Representation richness**: variance of activations at that layer ($\text{var}(A)$, log-transformed & z-scored).
- **CAM quality (non-monotonic)**: quadratic/absolute penalty for deviation from an optimal mid-range target entropy of $0.65$ ($-\text{abs}(H_{\text{norm}}(\text{CAM}) - 0.65)$, z-scored). This penalizes both degenerate uninformative uniform maps (entropy $\approx 1.0$) and single-pixel spike artifacts (entropy $\approx 0.0$).

Each metric was z-scored across layers and averaged to produce a combined layer score.
- **Winning layer**: `TCN.TCN.9.conv1d` (Block 9 of 24) — Combined Score: **+0.8931**, Mean Gradient: $1.96 \times 10^{-3}$, Activation Variance: $2.91$, CAM Entropy: $0.762$.
- Top runners-up: `TCN.TCN.11.conv1d` (+0.6865) and `TCN.TCN.15.conv1d` (+0.6148).

A visual comparison of CAM heatmaps across block depths ([cam_visual_comparison.png](results/layer_selection/cam_visual_comparison.png)) confirms that earlier blocks (e.g. Block 6) produce overly diffuse, blob-like attention maps, while mid-to-late blocks (e.g. Block 9 and Block 15) produce well-localized, temporally discriminative feature activations.

Full layer rankings and bar charts across all 24 blocks are saved in [layer_scores.json](results/layer_selection/layer_scores.json) and [layer_scores.png](results/layer_selection/layer_scores.png).

## Results

Statistical validation across **selection set ($N = 20$ pairs)**, **disjoint held-out set ($N = 15$ pairs)**, and **pooled dataset ($N = 35$ total pairs)** using target layer `TCN.TCN.9.conv1d` and freshly drawn independent random control maps per pair:

### VAD-Logit CAM Comparison

| Subset | $N$ Pairs | Real Speaker-vs-Speaker MAE | Real-vs-Random Control MAE | Wilcoxon $W$ | $p$-value | Rank-Biserial $r$ | Selection Bias Risk |
|---|---|---|---|---|---|---|---|
| **Selection Set** | $20$ | $0.2564 \pm 0.1094$ | $0.4090 \pm 0.0571$ | $17.0$ | $3.95 \times 10^{-4}$ | $0.838$ | Mild (Layer chosen on this set) |
| **Held-Out Set** | **$15$** | **$0.2580 \pm 0.0652$** | **$0.4185 \pm 0.0592$** | **$1.0$** | **$1.22 \times 10^{-4}$** | **$0.983$** | **None (Zero speaker overlap)** |
| **Pooled Total** | **$35$** | **$0.2571 \pm 0.0931$** | **$0.4131 \pm 0.0582$** | **$10.0$** | **$1.38 \times 10^{-7}$** | **$0.902$** | Minimal |

### Waveform CAM Comparison

| Subset | $N$ Pairs | Real Speaker-vs-Speaker MAE | Real-vs-Random Control MAE | Wilcoxon $W$ | $p$-value | Rank-Biserial $r$ | Selection Bias Risk |
|---|---|---|---|---|---|---|---|
| **Selection Set** | $20$ | $0.2497 \pm 0.0747$ | $0.4132 \pm 0.0523$ | $5.0$ | $1.91 \times 10^{-5}$ | $0.952$ | Mild (Layer chosen on this set) |
| **Held-Out Set** | **$15$** | **$0.2295 \pm 0.0640$** | **$0.4422 \pm 0.0554$** | **$0.0$** | **$6.10 \times 10^{-5}$** | **$1.000$** | **None (Zero speaker overlap)** |
| **Pooled Total** | **$35$** | **$0.2411 \pm 0.0710$** | **$0.4256 \pm 0.0555$** | **$1.0$** | **$8.15 \times 10^{-10}$** | **$0.981$** | Minimal |

### Key Artifacts & Visualizations

- **Selection Set Plot ($N=20$)**: [paired_comparison_plot.png](results/librispeech_gradcam/paired_comparison_plot.png)
- **Held-Out Set Plot ($N=15$)**: [holdout_paired_comparison_plot.png](results/librispeech_gradcam/holdout_paired_comparison_plot.png)
- **Pooled Total Plot ($N=35$)**: [pooled_paired_comparison_plot.png](results/librispeech_gradcam/pooled_paired_comparison_plot.png)
- **CSV Data**: [multi_pair_results.csv](results/librispeech_gradcam/multi_pair_results.csv) (selection set), [holdout_results.csv](results/librispeech_gradcam/holdout_results.csv) (held-out set), [pooled_results.csv](results/librispeech_gradcam/pooled_results.csv) (pooled set).
- **Layer Selection Artifacts**: [layer_scores.json](results/layer_selection/layer_scores.json), [layer_scores.png](results/layer_selection/layer_scores.png), [cam_visual_comparison.png](results/layer_selection/cam_visual_comparison.png).

The held-out validation confirms that the attention sensitivity effect is completely genuine and not an artifact of layer selection bias: on unseen, non-overlapping speakers, real speaker-vs-speaker MAE remains low ($0.229–0.258$), statistically significantly lower ($p < 0.0001$) than random control MAE ($0.418–0.442$), with an effect size of $r \ge 0.983$.

## Limitations / Next Steps

- **Sample Size & Diversity**: While extended to $N = 35$ total pairs across 70 distinct speakers, evaluations remain focused on clean speech mixtures from LibriSpeech. Noisy or reverberant environments should be tested in future work.
- **Single Target Layer**: Multi-pair and held-out validations fix the target layer to `TCN.TCN.9.conv1d` (the layer selected via Part A's ablation). Multi-layer or ensemble-based activation maps were not evaluated.
- **CPU Execution**: Verification and statistical benchmarks were conducted on CPU (`--device cpu`). GPU execution is supported via `--device cuda`.
