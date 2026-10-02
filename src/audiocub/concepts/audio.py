import pathlib

import librosa
import numpy
import pandas

THRESHOLDS = None
# audio sampling rate
SR = 16_000

CONCEPT_METADATA = {
    # Binarized Numeric Concepts (Bucketized)
    "peak_frequency::low": "binarized_numeric",
    "peak_frequency::medium": "binarized_numeric",
    "peak_frequency::high": "binarized_numeric",
    "spectral_centroid::low": "binarized_numeric",
    "spectral_centroid::medium": "binarized_numeric",
    "spectral_centroid::high": "binarized_numeric",
    "spectral_bandwidth::narrow": "binarized_numeric",
    "spectral_bandwidth::medium": "binarized_numeric",
    "spectral_bandwidth::wide": "binarized_numeric",
    "spectral_flatness::tonal": "binarized_numeric",
    "spectral_flatness::moderate": "binarized_numeric",
    "spectral_flatness::noisy": "binarized_numeric",
    "harmonic_ratio::low": "binarized_numeric",
    "harmonic_ratio::medium": "binarized_numeric",
    "harmonic_ratio::high": "binarized_numeric",
    "percussive_ratio::low": "binarized_numeric",
    "percussive_ratio::medium": "binarized_numeric",
    "percussive_ratio::high": "binarized_numeric",
    "syllable_rate::slow": "binarized_numeric",
    "syllable_rate::moderate": "binarized_numeric",
    "syllable_rate::rapid": "binarized_numeric",
    "note_duration::short_staccato": "binarized_numeric",
    "note_duration::medium": "binarized_numeric",
    "note_duration::long_sustained": "binarized_numeric",
    "f0_range::monotone": "binarized_numeric",
    "f0_range::moderate_sweep": "binarized_numeric",
    "f0_range::wide_sweep": "binarized_numeric",
    "rhythm::regular": "binarized_numeric",
    "rhythm::irregular": "binarized_numeric",
    "zero_crossing_rate::low": "binarized_numeric",
    "zero_crossing_rate::medium": "binarized_numeric",
    "zero_crossing_rate::high": "binarized_numeric",
    "spectral_rolloff::low": "binarized_numeric",
    "spectral_rolloff::medium": "binarized_numeric",
    "spectral_rolloff::high": "binarized_numeric",

    # Pure Binary Concepts
    "has_mechanical_drumming": "binary",
    "has_rhythmic_cooing": "binary",
    "has_guttural_croak_grunt": "binary",
    "has_rapid_rattle": "binary",
    "has_trill": "binary",
    "has_harmonic_stacking": "binary",
    "has_high_motif_variance": "binary",
    "has_wingbeat_hum": "binary",
}

import librosa
import numpy


def extract_all_audio_concepts(file_path: pathlib.Path) -> dict:
    audio, _ = librosa.load(file_path, sr=SR)

    return _extract_all_audio_concepts(audio)


def extract_all_audio_concepts_extended(file_path: pathlib.Path) -> dict:
    audio, _ = librosa.load(file_path, sr=SR)

    return _extract_all_audio_concepts_extended(audio)


def _extract_all_audio_concepts(audio: numpy.ndarray) -> dict:
    default_concepts = {
        "has_ultra_high_register": False,
        "has_deep_low_pitch": False,
        "has_buzzy_raspy_texture": False,
        "has_pure_tonal_whistle": False,
        "has_harmonic_stacking": False,
        "has_broad_bandwidth": False,
        "has_rapid_rattle": False,
        "has_trill": False,
        "has_frequency_sweep": False,
        "has_mechanical_drumming": False,
        "has_high_motif_variance": False,
        "has_guttural_croak_grunt": False,
        "has_rhythmic_cooing": False,
    }

    if audio is None or len(audio) == 0:
        return default_concepts

    try:
        duration = len(audio) / SR
        if duration < 0.1:
            return default_concepts

        # Fast normalization
        max_val = numpy.max(numpy.abs(audio))
        audio_norm = audio / max_val if max_val > 0 else audio

        # 1. Single STFT Pass (hop_length=1024 cuts frame matrix size in half)
        hop_length = 1024
        n_fft = 2048
        stft = numpy.abs(librosa.stft(audio_norm, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

        power_spec = numpy.sum(stft, axis=1)
        peak_freq = freqs[numpy.argmax(power_spec)] if len(power_spec) > 0 else 0.0

        # Vectorized Spectral Centroid
        stft_sum = numpy.sum(stft, axis=0) + 1e-8
        centroid_arr = numpy.dot(freqs, stft) / stft_sum
        mean_centroid = float(numpy.mean(centroid_arr))
        std_centroid = float(numpy.std(centroid_arr))

        # Vectorized Spectral Bandwidth
        freq_diff = freqs[:, None] - centroid_arr
        mean_bandwidth = float(numpy.mean(numpy.sqrt(numpy.sum((freq_diff ** 2) * stft, axis=0) / stft_sum)))

        # Vectorized Spectral Flatness
        log_stft = numpy.log(stft + 1e-8)
        gmean = numpy.exp(numpy.mean(log_stft, axis=0))
        amean = numpy.mean(stft, axis=0) + 1e-8
        mean_flatness = float(numpy.mean(gmean / amean))

        # 2. HPSS in Frequency Domain (Reduced kernel size)
        S_harm, S_perc = librosa.decompose.hpss(stft, kernel_size=(13, 13))
        total_energy = numpy.sum(stft ** 2) + 1e-8
        harmonic_ratio = float(numpy.sum(S_harm ** 2) / total_energy)
        percussive_ratio = float(numpy.sum(S_perc ** 2) / total_energy)

        # 3. Onset Envelopes from Pre-computed STFT
        onset_env = librosa.onset.onset_strength(S=stft, sr=SR, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=hop_length, units="time")
        onset_rate = len(onsets) / duration

        perc_env = librosa.onset.onset_strength(S=S_perc, sr=SR, hop_length=hop_length)
        perc_onsets = librosa.onset.onset_detect(onset_envelope=perc_env, sr=SR, hop_length=hop_length, units="time")
        perc_rate = len(perc_onsets) / duration

        inter_onset_std = 0.0
        if len(onsets) > 2:
            inter_onset_std = float(numpy.std(numpy.diff(onsets)))

        # 4. Instantaneous Frequency Sweep via STFT Bin Tracking (Replaces YIN/pYIN)
        f_mask = (freqs >= 300) & (freqs <= 8000)
        sub_stft = stft[f_mask, :]
        sub_freqs = freqs[f_mask]

        peak_mags = numpy.max(sub_stft, axis=0)
        peak_bins = numpy.argmax(sub_stft, axis=0)

        valid_mask = peak_mags > (0.10 * numpy.max(peak_mags))
        f0_delta = 0.0
        if numpy.sum(valid_mask) > 5:
            valid_f0 = sub_freqs[peak_bins[valid_mask]]
            f0_delta = float(numpy.max(valid_f0) - numpy.min(valid_f0))

        return {
            "has_ultra_high_register": bool(peak_freq >= 6000 or mean_centroid >= 5000),
            "has_deep_low_pitch": bool(mean_centroid <= 1500),
            "has_buzzy_raspy_texture": bool(mean_flatness >= 0.15 and harmonic_ratio < 0.40),
            "has_pure_tonal_whistle": bool(mean_flatness <= 0.02 and harmonic_ratio >= 0.60),
            "has_harmonic_stacking": bool(harmonic_ratio >= 0.55 and mean_flatness <= 0.08),
            "has_broad_bandwidth": bool(mean_bandwidth >= 3200),
            "has_rapid_rattle": bool(onset_rate >= 10.0),
            "has_trill": bool(onset_rate >= 7.0 and inter_onset_std < 0.035),
            "has_frequency_sweep": bool(f0_delta >= 1000.0),
            "has_mechanical_drumming": bool(perc_rate >= 8.0 and mean_centroid <= 2200 and percussive_ratio > 0.35),
            "has_high_motif_variance": bool(std_centroid >= 1200.0 and len(onsets) >= 4),
            "has_guttural_croak_grunt": bool(mean_centroid <= 1100 and mean_flatness >= 0.10),
            "has_rhythmic_cooing": bool(mean_centroid <= 1600 and 1.0 <= onset_rate <= 4.0 and inter_onset_std < 0.04),
        }

    except Exception:
        return default_concepts


def energy_low_freq_ratio(power_spec: numpy.ndarray, freqs: numpy.ndarray, cutoff: float = 800.0) -> float:
    """Helper function to calculate proportion of spectral energy below cutoff Hz."""
    low_idx = freqs <= cutoff
    total_energy = numpy.sum(power_spec) + 1e-8
    return float(numpy.sum(power_spec[low_idx]) / total_energy)


def _extract_all_audio_concepts_extended(audio: numpy.ndarray) -> dict:
    concept_keys = [
        "peak_frequency::low", "peak_frequency::medium", "peak_frequency::high",
        "spectral_centroid::low", "spectral_centroid::medium", "spectral_centroid::high",
        "spectral_bandwidth::narrow", "spectral_bandwidth::medium", "spectral_bandwidth::wide",
        "spectral_flatness::tonal", "spectral_flatness::moderate", "spectral_flatness::noisy",
        "harmonic_ratio::low", "harmonic_ratio::medium", "harmonic_ratio::high",
        "percussive_ratio::low", "percussive_ratio::medium", "percussive_ratio::high",
        "syllable_rate::slow", "syllable_rate::moderate", "syllable_rate::rapid",
        "note_duration::short_staccato", "note_duration::medium", "note_duration::long_sustained",
        "f0_range::monotone", "f0_range::moderate_sweep", "f0_range::wide_sweep",
        "rhythm::regular", "rhythm::irregular",
        "zero_crossing_rate::low", "zero_crossing_rate::medium", "zero_crossing_rate::high",
        "spectral_rolloff::low", "spectral_rolloff::medium", "spectral_rolloff::high",
        "has_mechanical_drumming", "has_rhythmic_cooing", "has_guttural_croak_grunt",
        "has_rapid_rattle", "has_trill", "has_harmonic_stacking",
        "has_high_motif_variance", "has_wingbeat_hum"
    ]
    concepts = {k: False for k in concept_keys}

    if audio is None or len(audio) == 0:
        return concepts

    try:
        duration = len(audio) / SR
        if duration < 0.1:
            return concepts

        # Fast normalization
        max_val = numpy.max(numpy.abs(audio))
        audio_norm = audio / max_val if max_val > 0 else audio

        # 1. Base STFT (hop_length=1024 halves matrix size and double-speeds execution)
        hop_length = 1024
        n_fft = 2048
        stft = numpy.abs(librosa.stft(audio_norm, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)

        # Power Spectrum
        power_spec = numpy.sum(stft, axis=1)
        peak_freq = freqs[numpy.argmax(power_spec)] if len(power_spec) > 0 else 0.0

        # Vectorized Centroid & Std
        stft_sum = numpy.sum(stft, axis=0) + 1e-8
        centroid_arr = numpy.dot(freqs, stft) / stft_sum
        centroid = float(numpy.mean(centroid_arr))
        std_centroid = float(numpy.std(centroid_arr))

        # Vectorized Bandwidth
        freq_diff = freqs[:, None] - centroid_arr
        bandwidth = float(numpy.mean(numpy.sqrt(numpy.sum((freq_diff ** 2) * stft, axis=0) / stft_sum)))

        # Vectorized Flatness
        log_stft = numpy.log(stft + 1e-8)
        gmean = numpy.exp(numpy.mean(log_stft, axis=0))
        amean = numpy.mean(stft, axis=0) + 1e-8
        flatness = float(numpy.mean(gmean / amean))

        # Direct NumPy Zero Crossing Rate (<1ms)
        zcr = float(numpy.mean(numpy.diff(numpy.signbit(audio_norm)) != 0))

        # Vectorized Rolloff (85%)
        cumsum_stft = numpy.cumsum(stft, axis=0)
        rolloff_idx = numpy.argmax(cumsum_stft >= (0.85 * cumsum_stft[-1, :]), axis=0)
        rolloff = float(numpy.mean(freqs[rolloff_idx]))

        # 2. HPSS with Reduced Kernel Size
        S_harm, S_perc = librosa.decompose.hpss(stft, kernel_size=(13, 13))
        total_energy = numpy.sum(stft ** 2) + 1e-8
        harmonic_ratio = float(numpy.sum(S_harm ** 2) / total_energy)
        percussive_ratio = float(numpy.sum(S_perc ** 2) / total_energy)

        # 3. Onset & Rhythm Tracking via Pre-computed Envelopes
        onset_env = librosa.onset.onset_strength(S=stft, sr=SR, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=hop_length, units="time")

        perc_env = librosa.onset.onset_strength(S=S_perc, sr=SR, hop_length=hop_length)
        perc_onsets = librosa.onset.onset_detect(onset_envelope=perc_env, sr=SR, hop_length=hop_length, units="time")

        syllable_rate = len(onsets) / duration
        perc_rate = len(perc_onsets) / duration
        mean_duration = duration / max(1, len(onsets))

        inter_onset_std = 0.0
        if len(onsets) > 2:
            inter_onset_std = float(numpy.std(numpy.diff(onsets)))

        # 4. Instantaneous f0_delta via STFT Bin Tracking (Replaces pyin/yin)
        f_mask = (freqs >= 300) & (freqs <= 8000)
        sub_stft = stft[f_mask, :]
        sub_freqs = freqs[f_mask]

        peak_mags = numpy.max(sub_stft, axis=0)
        peak_bins = numpy.argmax(sub_stft, axis=0)

        valid_mask = peak_mags > (0.10 * numpy.max(peak_mags))
        f0_delta = 0.0
        if numpy.sum(valid_mask) > 5:
            valid_f0 = sub_freqs[peak_bins[valid_mask]]
            f0_delta = float(numpy.max(valid_f0) - numpy.min(valid_f0))

        # ---------------------------------------------------------------------
        # Binarize Numeric Concepts (Bucketing)
        # ---------------------------------------------------------------------
        concepts["peak_frequency::low"] = peak_freq < 2000
        concepts["peak_frequency::medium"] = 2000 <= peak_freq <= 5000
        concepts["peak_frequency::high"] = peak_freq > 5000

        concepts["spectral_centroid::low"] = centroid < 1500
        concepts["spectral_centroid::medium"] = 1500 <= centroid <= 3500
        concepts["spectral_centroid::high"] = centroid > 3500

        concepts["spectral_bandwidth::narrow"] = bandwidth < 1500
        concepts["spectral_bandwidth::medium"] = 1500 <= bandwidth <= 3500
        concepts["spectral_bandwidth::wide"] = bandwidth > 3500

        concepts["spectral_flatness::tonal"] = flatness < 0.03
        concepts["spectral_flatness::moderate"] = 0.03 <= flatness <= 0.15
        concepts["spectral_flatness::noisy"] = flatness > 0.15

        concepts["harmonic_ratio::low"] = harmonic_ratio < 0.35
        concepts["harmonic_ratio::medium"] = 0.35 <= harmonic_ratio <= 0.65
        concepts["harmonic_ratio::high"] = harmonic_ratio > 0.65

        concepts["percussive_ratio::low"] = percussive_ratio < 0.15
        concepts["percussive_ratio::medium"] = 0.15 <= percussive_ratio <= 0.35
        concepts["percussive_ratio::high"] = percussive_ratio > 0.35

        concepts["syllable_rate::slow"] = syllable_rate < 3.0
        concepts["syllable_rate::moderate"] = 3.0 <= syllable_rate <= 8.0
        concepts["syllable_rate::rapid"] = syllable_rate > 8.0

        concepts["note_duration::short_staccato"] = mean_duration < 0.05
        concepts["note_duration::medium"] = 0.05 <= mean_duration <= 0.30
        concepts["note_duration::long_sustained"] = mean_duration > 0.30

        concepts["f0_range::monotone"] = f0_delta < 300.0
        concepts["f0_range::moderate_sweep"] = 300.0 <= f0_delta <= 1200.0
        concepts["f0_range::wide_sweep"] = f0_delta > 1200.0

        concepts["rhythm::regular"] = inter_onset_std < 0.03
        concepts["rhythm::irregular"] = inter_onset_std >= 0.03

        concepts["zero_crossing_rate::low"] = zcr < 0.05
        concepts["zero_crossing_rate::medium"] = 0.05 <= zcr <= 0.15
        concepts["zero_crossing_rate::high"] = zcr > 0.15

        concepts["spectral_rolloff::low"] = rolloff < 3000
        concepts["spectral_rolloff::medium"] = 3000 <= rolloff <= 7000
        concepts["spectral_rolloff::high"] = rolloff > 7000

        # ---------------------------------------------------------------------
        # Inherently Binary Concepts
        # ---------------------------------------------------------------------
        concepts["has_mechanical_drumming"] = bool(perc_rate >= 8.0 and centroid <= 2200 and percussive_ratio > 0.35)
        concepts["has_rhythmic_cooing"] = bool(
            centroid <= 1600 and 1.0 <= syllable_rate <= 4.0 and inter_onset_std < 0.04)
        concepts["has_guttural_croak_grunt"] = bool(centroid <= 1100 and flatness >= 0.10)
        concepts["has_rapid_rattle"] = bool(syllable_rate >= 10.0 and bandwidth >= 3000)
        concepts["has_trill"] = bool(syllable_rate >= 7.0 and inter_onset_std < 0.035)
        concepts["has_harmonic_stacking"] = bool(harmonic_ratio >= 0.55 and flatness <= 0.08)
        concepts["has_high_motif_variance"] = bool(std_centroid >= 1200.0 and len(onsets) >= 4)

        low_idx = freqs <= 800.0
        tot_e = numpy.sum(power_spec) + 1e-8
        low_ratio = float(numpy.sum(power_spec[low_idx]) / tot_e)
        concepts["has_wingbeat_hum"] = bool(centroid <= 800 and low_ratio > 0.60)

    except Exception:
        pass

    return concepts

