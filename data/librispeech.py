"""
Minimal LibriSpeech indexer/loader. Works with any subset directory
(train-clean-100, train-clean-360, dev-clean, test-clean, ...) laid out in
the standard structure:
  <root>/<speaker_id>/<chapter_id>/<speaker_id>-<chapter_id>-<utterance_id>.flac
Only audio + speaker id are used; transcripts are ignored.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from data.config import SAMPLE_RATE, MIN_UTTERANCE_S, MAX_UTTERANCE_S


def index_librispeech(root: str) -> Dict[str, List[Path]]:
    """Returns {speaker_id: [utterance_path, ...]}."""
    root = Path(root)
    speakers: Dict[str, List[Path]] = {}
    for ext in ("*.flac", "*.wav"):
        for p in root.rglob(ext):
            speaker_id = p.name.split("-")[0]
            speakers.setdefault(speaker_id, []).append(p)
        if speakers:
            break
    if not speakers:
        raise FileNotFoundError(
            f"No .flac/.wav files found under {root} -- check the path points "
            f"at a LibriSpeech subset directory (e.g. .../train-clean-100)."
        )
    return speakers


def load_utterance(path: Path) -> np.ndarray:
    """Loads audio, mono float32, resampled to SAMPLE_RATE if necessary."""
    wav, fs = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if fs != SAMPLE_RATE:
        # LibriSpeech ships at 16 kHz already; this is just a safety net.
        import resampy
        wav = resampy.resample(wav, fs, SAMPLE_RATE)
    return wav.astype(np.float32)


def utterance_duration_s(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


class SpeakerSampler:
    """Samples two distinct speakers per mixture, plus a separate
    auxiliary/anchor utterance for whichever speaker is the target,
    filtered to a plausible utterance duration range."""

    def __init__(self, root: str, seed: int = 0,
                 min_s: float = MIN_UTTERANCE_S, max_s: float = MAX_UTTERANCE_S):
        self.speakers = index_librispeech(root)
        self.speaker_ids = [s for s, utts in self.speakers.items() if len(utts) >= 2]
        if len(self.speaker_ids) < 2:
            raise ValueError(
                "Need at least 2 speakers with >=2 utterances each "
                "(one for the mixture, one for the anchor)."
            )
        self.rng = random.Random(seed)
        self.min_s = min_s
        self.max_s = max_s

    def sample_two_speakers(self):
        return self.rng.sample(self.speaker_ids, 2)

    def sample_utterance(self, speaker_id: str, exclude: Optional[Path] = None,
                          max_tries: int = 20) -> Path:
        """Samples an utterance for `speaker_id` within [min_s, max_s]
        seconds, optionally excluding one path (used to keep the anchor
        utterance different from the one used in the mixture itself)."""
        candidates = self.speakers[speaker_id]
        if exclude is not None and len(candidates) > 1:
            candidates = [c for c in candidates if c != exclude]

        best = None
        for _ in range(max_tries):
            cand = self.rng.choice(candidates)
            dur = utterance_duration_s(cand)
            if self.min_s <= dur <= self.max_s:
                return cand
            best = cand  # fallback if nothing in range after max_tries
        return best
