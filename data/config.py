"""Minimal config for LibriSpeech loading, used only by data/librispeech.py.

Values match the LibriSpeech corpus itself (16 kHz) and the observed
utterance duration range used when sampling real mixtures.
"""

SAMPLE_RATE = 16000
MIN_UTTERANCE_S = 3.0
MAX_UTTERANCE_S = 16.0
