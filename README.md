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
python scripts/verify_librispeech_gradcam.py --librispeech-root data/librispeech_samples --device cpu
```

This will:
1. Sample two distinct real speakers from `--librispeech-root` and mix them.
2. Run the model forward pass and compute Grad-CAM at a mid-depth TCN layer (`TCN.TCN.12.conv1d`) for two targets per
   speaker: the pre-sigmoid VAD logit and the separated waveform.
3. Print the source filenames/speaker IDs, target values, and three comparison metrics (see
   [Methodology](#methodology)).
4. Save four-panel figures and `metadata.json` (source filenames + speaker IDs + target layer, no absolute local
   paths) to `results/librispeech_gradcam/final/`.

## Methodology

Two adaptation decisions mattered and are worth calling out explicitly:

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
VAD-logit CAM, speaker vs speaker: MAE ≈ 0.23–0.25 (max diff ≈ 1.000)
VAD-logit CAM, speaker vs random noise: MAE ≈ 0.46 (max diff ≈ 0.995)
```

The max-diff numbers alone would suggest the real-vs-real and real-vs-random comparisons are equally "different."
The MAE numbers show the real speaker-vs-speaker CAMs are noticeably more self-similar to each other than either is
to random noise — a genuine (if modest) signal that max-diff alone would have missed.

## Results

See `results/librispeech_gradcam/final/`:

- `example_speaker0.png`, `example_speaker1.png` — spectrogram | VAD-logit CAM | waveform CAM | overlay, one figure
  per target speaker, computed from the same real 2-speaker LibriSpeech mixture.
- `source_speakers_comparison.jpg` — the two pre-mix source utterances' spectrograms side by side, confirming they
  are genuinely different real voices (waveform correlation ≈ −0.002), not a duplicated file.
- `metadata.json` — source filenames, speaker IDs, and target layer for traceability.

**Finding** (single speaker pair, see [Limitations](#limitations--next-steps)):

| Comparison | Independent min-max MAE | Shared-scale MAE | vs. random-noise MAE |
|---|---|---|---|
| VAD-logit CAM, speaker 0 vs. speaker 1 | 0.253 | 0.228 | 0.457 |
| Waveform CAM, speaker 0 vs. speaker 1 | 0.253 | 0.229 | 0.491 |

<img width="3317" height="767" alt="example_speaker0" src="https://github.com/user-attachments/assets/b05e322b-2459-4c48-ac1a-71b000f80c25" />
<img width="3317" height="767" alt="example_speaker1" src="https://github.com/user-attachments/assets/58b4d542-ab32-420d-9c0d-69fa363d828e" />

Real speaker-vs-speaker CAMs are consistently more self-similar (lower MAE) than real-vs-random-noise. This is a
promising initial result, not a settled conclusion — see limitations below.

## Limitations / Next Steps

- **n = 1 speaker pair.** All numbers above come from a single mixture of two speakers. No statistical significance
  testing has been done; the MAE gap could be pair-specific.
- **Single mid-depth TCN layer only** (`TCN.TCN.12.conv1d`, layer 12 of 24). Other layers/depths, or averaging across
  layers, aren't explored here.
- **CPU-only verified.** GPU execution should work (`--device cuda`) but hasn't been separately validated in this repo.
- Extending to more speaker pairs, more layers, and a proper significance test (e.g. bootstrap over many mixtures)
  would be the natural next step before treating the MAE gap as a real finding rather than a promising signal.
