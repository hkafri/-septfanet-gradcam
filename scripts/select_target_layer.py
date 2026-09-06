"""Select optimal Grad-CAM target layer among all TCN conv1d blocks.

Scores candidate layers on 5 speaker pairs using 3 criteria:
1. Gradient signal strength (mean absolute gradient)
2. Representation richness (activation variance)
3. CAM quality (normalized entropy of heatmap)

Artifacts saved:
- results/layer_selection/layer_scores.json
- results/layer_selection/layer_scores.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAMHook


SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
NUM_PAIRS = 5


def normalize_audio(audio):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    return audio / max(peak, 1e-8) * 0.9


def prepare_mixture(paths):
    target_length = int(TARGET_SECONDS * SAMPLE_RATE)
    signals = []
    for path in paths:
        signal = load_utterance(path)[:target_length]
        padded = np.zeros(target_length, dtype=np.float32)
        padded[:len(signal)] = signal
        signals.append(padded)
    mixture = normalize_audio(signals[0] + signals[1])
    return torch.from_numpy(mixture).unsqueeze(0)


def compute_normalized_entropy(cam_1d):
    """Compute normalized Shannon entropy of 1D CAM array in [0, 1]."""
    if cam_1d.max() <= 1e-12:
        return 0.0
    p = (cam_1d + 1e-12) / (cam_1d.sum() + 1e-12 * len(cam_1d))
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(cam_1d))
    return float(entropy / max_entropy)


def evaluate_candidate_layers(model, candidate_layers, pairs, device):
    layer_metrics = {
        layer: {"grad_strength": [], "rep_richness": [], "cam_quality": []}
        for layer in candidate_layers
    }

    for pair_idx, paths in enumerate(pairs):
        audio = prepare_mixture(paths)
        for speaker_idx in range(2):
            input_audio = audio.clone().detach().to(device).float().requires_grad_(True)
            
            # Forward pass to get VAD logits
            with torch.enable_grad():
                _ = model(input_audio)
                vad_logits = model.vad_logits
                target_frame = int(torch.argmax(vad_logits[0, speaker_idx].abs()))
                target = vad_logits[0, speaker_idx, target_frame]

            # Evaluate each layer
            for layer_name in candidate_layers:
                hook = GradCAMHook(model, layer_name)
                try:
                    input_audio_copy = audio.clone().detach().to(device).float().requires_grad_(True)
                    with torch.enable_grad():
                        _ = model(input_audio_copy)
                        target_val = model.vad_logits[0, speaker_idx, target_frame]
                        target_val.backward(retain_graph=True)

                    act = hook.activations.detach().cpu().numpy()[0]  # (C, T)
                    grad = hook.gradients.detach().cpu().numpy()[0]   # (C, T)

                    # 1. Gradient signal strength
                    grad_strength = float(np.mean(np.abs(grad)))

                    # 2. Representation richness (activation variance)
                    rep_richness = float(np.var(act))

                    # 3. CAM quality (normalized entropy)
                    weights = np.mean(grad, axis=1, keepdims=True)     # (C, 1)
                    cam = np.maximum(np.sum(weights * act, axis=0), 0)  # (T,)
                    cam_quality = compute_normalized_entropy(cam)

                    layer_metrics[layer_name]["grad_strength"].append(grad_strength)
                    layer_metrics[layer_name]["rep_richness"].append(rep_richness)
                    layer_metrics[layer_name]["cam_quality"].append(cam_quality)
                finally:
                    hook.remove_hooks()

    return layer_metrics


def score_and_rank_layers(layer_metrics):
    layers = list(layer_metrics.keys())
    
    # Average across samples per layer
    avg_grad = np.array([np.mean(layer_metrics[l]["grad_strength"]) for l in layers])
    avg_rep = np.array([np.mean(layer_metrics[l]["rep_richness"]) for l in layers])
    avg_cam = np.array([np.mean(layer_metrics[l]["cam_quality"]) for l in layers])

    # Log-transform positive quantities before z-scoring
    log_grad = np.log(avg_grad + 1e-12)
    log_rep = np.log(avg_rep + 1e-12)

    # Z-scores
    z_grad = (log_grad - np.mean(log_grad)) / (np.std(log_grad) + 1e-8)
    z_rep = (log_rep - np.mean(log_rep)) / (np.std(log_rep) + 1e-8)
    z_cam = (avg_cam - np.mean(avg_cam)) / (np.std(avg_cam) + 1e-8)

    combined_scores = (z_grad + z_rep + z_cam) / 3.0

    rankings = []
    for idx, layer in enumerate(layers):
        block_num = int(layer.split(".")[2]) if "TCN.TCN." in layer else idx
        rankings.append({
            "layer": layer,
            "block_index": block_num,
            "combined_score": float(combined_scores[idx]),
            "z_grad": float(z_grad[idx]),
            "z_rep": float(z_rep[idx]),
            "z_cam": float(z_cam[idx]),
            "avg_grad_strength": float(avg_grad[idx]),
            "avg_rep_richness": float(avg_rep[idx]),
            "avg_cam_quality": float(avg_cam[idx]),
        })

    # Sort descending by score, tie-break preferring later block_index if scores are within 0.02
    rankings.sort(key=lambda x: (x["combined_score"], x["block_index"]), reverse=True)

    return rankings


def plot_layer_scores(rankings, output_path):
    # Sort by block index for clean 0 -> 23 visualization
    by_block = sorted(rankings, key=lambda x: x["block_index"])
    blocks = [x["block_index"] for x in by_block]
    scores = [x["combined_score"] for x in by_block]

    winner = max(rankings, key=lambda x: x["combined_score"])

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    colors = ["darkorange" if b == winner["block_index"] else "steelblue" for b in blocks]
    bars = ax.bar([f"Block {b}" for b in blocks], scores, color=colors, edgecolor="black")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"TCN Layer Selection Scores (Winner: {winner['layer']}, Score: {winner['combined_score']:.3f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("TCN Block Index")
    ax.set_ylabel("Combined Z-Score (Grad + Rep + CAM Quality)")
    ax.tick_params(axis="x", rotation=45)

    # Highlight winner
    ax.annotate(
        f"Winner: {winner['layer']}",
        xy=(f"Block {winner['block_index']}", winner["combined_score"]),
        xytext=(winner["block_index"], winner["combined_score"] + 0.3),
        arrowprops=dict(facecolor="darkorange", shrink=0.05, width=1.5, headwidth=6),
        fontweight="bold",
        color="darkorange",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    device = "cpu"
    print("[*] Performing Principled Layer Selection...")

    sampler = SpeakerSampler("data/librispeech_samples", seed=42)
    speaker_ids = sampler.speaker_ids
    if len(speaker_ids) < NUM_PAIRS * 2:
        raise ValueError(f"Need at least {NUM_PAIRS * 2} speakers in data/librispeech_samples")

    pairs = []
    for i in range(NUM_PAIRS):
        spk1, spk2 = speaker_ids[2 * i], speaker_ids[2 * i + 1]
        p1 = sampler.sample_utterance(spk1)
        p2 = sampler.sample_utterance(spk2)
        pairs.append([p1, p2])

    config_path = gradcam_root / "configs" / "config_with_vad.json"
    checkpoint_path = gradcam_root / "weights" / "model_with_vad.pth"
    config = json.loads(config_path.read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)

    candidate_layers = [f"TCN.TCN.{i}.conv1d" for i in range(24)]

    print(f"[*] Evaluating {len(candidate_layers)} candidate layers over {NUM_PAIRS} speaker pairs...")
    layer_metrics = evaluate_candidate_layers(model, candidate_layers, pairs, device)
    rankings = score_and_rank_layers(layer_metrics)

    winning_layer = rankings[0]["layer"]
    print(f"\n[+] Layer Selection Results (Top 5):")
    for r in rankings[:5]:
        print(f"    Rank: {r['layer']:<20} Score: {r['combined_score']:+.4f} (Grad: {r['avg_grad_strength']:.2e}, RepVar: {r['avg_rep_richness']:.2f}, CAMEnt: {r['avg_cam_quality']:.3f})")

    output_dir = gradcam_root / "results" / "layer_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "layer_scores.json"
    json_path.write_text(json.dumps({"winning_layer": winning_layer, "rankings": rankings}, indent=2))

    plot_path = output_dir / "layer_scores.png"
    plot_layer_scores(rankings, plot_path)

    print(f"[+] Saved artifacts to:\n    - {json_path}\n    - {plot_path}")


if __name__ == "__main__":
    main()
