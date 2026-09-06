"""Fetch a disjoint held-out set of 30 distinct LibriSpeech speakers (15 pairs).

Ensures zero speaker overlap with the 40 speakers in data/librispeech_samples/.
Saves audio under data/librispeech_holdout/.
"""

import io
import sys
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

ORIGINAL_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"


def get_original_speaker_ids():
    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"Original samples directory does not exist: {ORIGINAL_DIR}")
    flac_files = list(ORIGINAL_DIR.glob("*.flac"))
    orig_spks = set(f.name.split("-")[0] for f in flac_files)
    return orig_spks


def fetch_holdout_samples(num_speakers=30, utterances_per_speaker=2):
    orig_spks = get_original_speaker_ids()
    print(f"[*] Found {len(orig_spks)} original speaker IDs to exclude.")

    print(f"[*] Streaming openslr/librispeech_asr (clean, validation split) for {num_speakers} new speakers...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    seen_speakers = {}
    for example in ds:
        speaker_id = str(example["speaker_id"])
        if speaker_id in orig_spks:
            continue
        bucket = seen_speakers.setdefault(speaker_id, [])
        if len(bucket) < utterances_per_speaker:
            bucket.append(example)
        if len(seen_speakers) == num_speakers and all(len(v) == utterances_per_speaker for v in seen_speakers.values()):
            break

    if len(seen_speakers) < num_speakers or any(len(v) < utterances_per_speaker for v in seen_speakers.values()):
        raise RuntimeError(
            f"Could not collect {utterances_per_speaker} utterances for {num_speakers} distinct holdout speakers "
            f"(got: { {k: len(v) for k, v in seen_speakers.items()} })"
        )

    holdout_spks = set(seen_speakers.keys())
    overlap = orig_spks.intersection(holdout_spks)
    print(f"[+] Holdout speakers collected: {len(holdout_spks)}")
    print(f"[+] Overlap with original 40 speakers: {len(overlap)}")
    assert len(overlap) == 0, f"Error: Speaker overlap detected: {overlap}"

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for speaker_id, examples in seen_speakers.items():
        for example in examples:
            audio_bytes = example["audio"]["bytes"]
            array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            utterance_id = example["id"]
            out_path = HOLDOUT_DIR / f"{utterance_id}.flac"
            sf.write(out_path, array, sr, format="FLAC")
            saved.append((speaker_id, utterance_id, out_path))

    print(f"[+] Saved {len(saved)} holdout files to {HOLDOUT_DIR}")
    return saved, orig_spks, holdout_spks


def main():
    saved, orig_spks, holdout_spks = fetch_holdout_samples(num_speakers=30)
    print("\n" + "=" * 60)
    print("HOLDOUT SET DISJOINTNESS VERIFICATION:")
    print(f"  Original speaker count: {len(orig_spks)}")
    print(f"  Holdout speaker count:  {len(holdout_spks)}")
    print(f"  Zero overlap verified:  {len(orig_spks.intersection(holdout_spks)) == 0}")
    print("=" * 60)


if __name__ == "__main__":
    main()
