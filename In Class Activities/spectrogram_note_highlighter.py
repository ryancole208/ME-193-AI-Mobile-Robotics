"""
Record 5 seconds of audio from the microphone, compute a spectrogram, and
highlight the loudest frequency for each sustained musical note on the plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import sounddevice as sd

# ---- Recording settings ----
DURATION = 5.0       # seconds to record
SAMPLE_RATE = 44100  # Hz

# ---- STFT settings ----
WINDOW_SIZE = 4096              # samples per FFT window (~93 ms): good frequency resolution
HOP_SIZE = 512                  # samples between windows (~12 ms): good time resolution
FREQ_MIN = 60.0                 # Hz, ignore rumble/hum below this
FREQ_MAX = 4000.0               # Hz, highest frequency considered for note detection
SILENCE_DB_BELOW_PEAK = 35.0    # frames quieter than (loudest frame - this) count as silence

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def record_audio(duration, sample_rate):
    print(f"Recording for {duration:.0f} seconds...")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float64")
        sd.wait()
    except sd.PortAudioError as exc:
        raise RuntimeError(
            "Could not record audio. Check that a microphone is connected and "
            "selected as the default input device."
        ) from exc
    print("Recording complete.")
    return audio[:, 0]


def compute_stft(signal, sample_rate, window_size, hop_size):
    """Short-time Fourier transform: returns (freqs, times, magnitude_spectrogram)."""
    window = np.hanning(window_size)
    n_frames = 1 + (len(signal) - window_size) // hop_size
    if n_frames < 1:
        raise ValueError("Recording too short for the chosen window size.")

    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    spec = np.empty((window_size // 2 + 1, n_frames), dtype=np.float64)
    times = np.empty(n_frames, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop_size
        frame = signal[start:start + window_size] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))
        times[i] = (start + window_size / 2) / sample_rate

    return freqs, times, spec


def freq_to_note(freq):
    """Nearest equal-tempered note name (A4 = 440 Hz), e.g. 'A4'."""
    midi_round = int(np.round(69 + 12 * np.log2(freq / 440.0)))
    name = NOTE_NAMES[midi_round % 12]
    octave = midi_round // 12 - 1
    return f"{name}{octave}"


def parabolic_peak_index(mag, i):
    """Sub-bin peak location via parabolic interpolation around bin i."""
    if i <= 0 or i >= len(mag) - 1:
        return float(i)
    y0, y1, y2 = mag[i - 1], mag[i], mag[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(i)
    return i + 0.5 * (y0 - y2) / denom


def find_peak_frequencies(freqs, spec, freq_min, freq_max):
    """For every time frame, find the loudest frequency within [freq_min, freq_max]."""
    band_idx = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    freq_resolution = freqs[1] - freqs[0]

    n_frames = spec.shape[1]
    peak_freqs = np.zeros(n_frames)
    peak_mags_db = np.zeros(n_frames)

    for i in range(n_frames):
        col = spec[:, i]
        bin_idx = band_idx[np.argmax(col[band_idx])]
        interp_bin = np.clip(parabolic_peak_index(col, bin_idx), 0, len(freqs) - 1)
        peak_freqs[i] = interp_bin * freq_resolution
        peak_mags_db[i] = 20 * np.log10(col[bin_idx] + 1e-12)

    return peak_freqs, peak_mags_db


def group_into_notes(times, peak_freqs, peak_mags_db, silence_threshold_db):
    """Merge consecutive frames that share a note into segments, and pick the
    loudest frame within each segment as the point to highlight."""
    notes = [freq_to_note(f) if f > 0 else None for f in peak_freqs]
    is_silent = peak_mags_db < silence_threshold_db

    segments = []
    start = 0
    for i in range(1, len(notes) + 1):
        boundary = i == len(notes) or notes[i] != notes[start] or is_silent[i] != is_silent[start]
        if boundary:
            if not is_silent[start] and notes[start] is not None:
                loudest_idx = start + int(np.argmax(peak_mags_db[start:i]))
                segments.append({
                    "note": notes[start],
                    "time": times[loudest_idx],
                    "freq": peak_freqs[loudest_idx],
                })
            start = i
    return segments


def plot_spectrogram(times, freqs, spec, segments, freq_max):
    spec_db = 20 * np.log10(spec + 1e-12)
    vmax = spec_db.max()
    vmin = vmax - 80  # 80 dB dynamic range

    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(times, freqs, spec_db, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_ylim(0, freq_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram — loudest frequency per note highlighted")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Magnitude (dB)")

    if segments:
        seg_times = [s["time"] for s in segments]
        seg_freqs = [s["freq"] for s in segments]
        ax.scatter(seg_times, seg_freqs, s=70, facecolors="white", edgecolors="black",
                   linewidths=1.2, zorder=5, label="Loudest frequency per note")

        label_outline = [pe.withStroke(linewidth=2.5, foreground="black")]
        for s in segments:
            ax.annotate(
                f"{s['note']}\n{s['freq']:.0f} Hz",
                xy=(s["time"], s["freq"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="white",
                fontweight="bold",
                path_effects=label_outline,
            )
        ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def main():
    audio = record_audio(DURATION, SAMPLE_RATE)
    freqs, times, spec = compute_stft(audio, SAMPLE_RATE, WINDOW_SIZE, HOP_SIZE)
    peak_freqs, peak_mags_db = find_peak_frequencies(freqs, spec, FREQ_MIN, FREQ_MAX)

    silence_threshold = peak_mags_db.max() - SILENCE_DB_BELOW_PEAK
    segments = group_into_notes(times, peak_freqs, peak_mags_db, silence_threshold)

    if segments:
        print("Detected notes:")
        for s in segments:
            print(f"  t={s['time']:.2f}s  {s['note']:<4} ({s['freq']:.1f} Hz)")
    else:
        print("No clear notes detected above the silence threshold.")

    fig = plot_spectrogram(times, freqs, spec, segments, FREQ_MAX)
    fig.savefig("spectrogram_output.png", dpi=150)
    print("Saved plot to spectrogram_output.png")
    plt.show()


if __name__ == "__main__":
    main()
