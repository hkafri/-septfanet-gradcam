"""Fetch 2 real LibriSpeech utterances (different speakers) without downloading
the full corpus, preferring HF streaming and falling back to torchaudio's full
download. Never falls back to synthetic audio.
"""

import sys
from pathlib import Path

import soundfile as sf

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "librispeech_samples"


def fetch_via_hf_streaming(output_dir, utterances_per_speaker=2):
    import io

    from datasets import Audio, load_dataset

    print("[*] Streaming openslr/librispeech_asr (clean, test split) from Hugging Face...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
    # Keep audio as raw encoded bytes; decoding via torchcodec is unavailable/unreliable
    # on this machine (missing FFmpeg shared libs), so decode manually with soundfile.
    ds = ds.cast_column("audio", Audio(decode=False))

    # SpeakerSampler requires >=2 utterances/speaker (one for the mixture, one
    # for the anchor), so collect that many per distinct speaker.
    seen_speakers = {}
    for example in ds:
        speaker_id = example["speaker_id"]
        bucket = seen_speakers.setdefault(speaker_id, [])
        if len(bucket) < utterances_per_speaker:
            bucket.append(example)
        if len(seen_speakers) == 2 and all(len(v) == utterances_per_speaker for v in seen_speakers.values()):
            break

    if len(seen_speakers) < 2 or any(len(v) < utterances_per_speaker for v in seen_speakers.values()):
        raise RuntimeError(
            f"Could not collect {utterances_per_speaker} utterances for 2 distinct speakers "
            f"(got: { {k: len(v) for k, v in seen_speakers.items()} })"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for speaker_id, examples in seen_speakers.items():
        for example in examples:
            audio_bytes = example["audio"]["bytes"]
            array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            utterance_id = example["id"]
            out_path = output_dir / f"{utterance_id}.flac"
            sf.write(out_path, array, sr, format="FLAC")
            saved.append((speaker_id, utterance_id, out_path))
    return saved


def fetch_via_torchaudio(output_dir, utterances_per_speaker=2):
    import torchaudio

    print("[*] Falling back to torchaudio.datasets.LIBRISPEECH full download (test-clean)...")
    download_root = output_dir.parent / "librispeech_torchaudio_download"
    download_root.mkdir(parents=True, exist_ok=True)
    dataset = torchaudio.datasets.LIBRISPEECH(root=str(download_root), url="test-clean", download=True)

    # SpeakerSampler requires >=2 utterances/speaker (one for the mixture, one
    # for the anchor), so collect that many per distinct speaker.
    seen_speakers = {}
    for i in range(len(dataset)):
        waveform, sr, transcript, speaker_id, chapter_id, utterance_id = dataset[i]
        bucket = seen_speakers.setdefault(speaker_id, [])
        if len(bucket) < utterances_per_speaker:
            bucket.append((waveform, sr, speaker_id, chapter_id, utterance_id))
        if len(seen_speakers) == 2 and all(len(v) == utterances_per_speaker for v in seen_speakers.values()):
            break

    if len(seen_speakers) < 2 or any(len(v) < utterances_per_speaker for v in seen_speakers.values()):
        raise RuntimeError(
            f"Could not collect {utterances_per_speaker} utterances for 2 distinct speakers "
            f"(got: { {k: len(v) for k, v in seen_speakers.items()} })"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for speaker_id, entries in seen_speakers.items():
        for waveform, sr, spk, chapter_id, utterance_id in entries:
            name = f"{spk}-{chapter_id}-{utterance_id:04d}"
            out_path = output_dir / f"{name}.flac"
            sf.write(out_path, waveform.squeeze(0).numpy(), sr, format="FLAC")
            saved.append((speaker_id, name, out_path))
    return saved


def main():
    try:
        saved = fetch_via_hf_streaming(OUTPUT_DIR)
        source = "huggingface streaming"
    except Exception as e:
        print(f"[!] HF streaming failed: {e}")
        saved = fetch_via_torchaudio(OUTPUT_DIR)
        source = "torchaudio full download"

    print(f"\n[+] Source: {source}")
    print("[+] Saved real LibriSpeech utterances:")
    for speaker_id, utterance_id, path in saved:
        assert path.exists(), f"File was not actually written: {path}"
        info = sf.info(str(path))
        print(f"    speaker_id={speaker_id}  utterance_id={utterance_id}  "
              f"path={path.resolve()}  duration={info.frames / info.samplerate:.2f}s")


if __name__ == "__main__":
    main()
