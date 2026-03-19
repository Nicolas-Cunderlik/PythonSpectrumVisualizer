# This script is a better version of the original.
# It plays the audio file at the same time as the visualizer.
# Originally, it was supposed to use real-time system audio, but on MacOS this takes extra setup.

# The script:
# 1. Loads audio files using librosa (mp3 now, since wav is too big for full songs)
# 2. Computes the FFT for the whole audio (intial load takes a while)
# 3. Plays the audio using sounddevice, with a callback to stream the audio in chunks
# 4. Updates the visualizer in sync with the audio playback, using the precomputed FFT data

# This script still does not allow for real-time audio input, which I will probably
# just have to do in C++ to avoid the additional setup.

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa

# Constants and audio config
FILE = "115bpm-c#min-juicewrldbeat-mastered.mp3"
BUFFER_SIZE = 1024
FPS = 30

audio, sample_rate = librosa.load(FILE, sr=None, mono=True)
audio *= 0.5  # KEEP THIS OR IT WILL BLOW YOUR EARS OUT

audio_len = len(audio)

# ==============================================================================
# Precompute FFT for the entire audio file
# ==============================================================================
window = np.hanning(BUFFER_SIZE)
freqs = np.fft.rfftfreq(BUFFER_SIZE, 1 / sample_rate)

num_chunks = int(np.ceil(audio_len / BUFFER_SIZE))
fft_chunks = []

for i in range(num_chunks):
    start = i * BUFFER_SIZE
    end = start + BUFFER_SIZE
    buf = audio[start:end]
    if len(buf) < BUFFER_SIZE:
        buf = np.pad(buf, (0, BUFFER_SIZE - len(buf)))
    buf = buf * window
    fft = np.fft.rfft(buf)
    mag = np.abs(fft)
    mag = np.maximum(mag, 1e-10)
    db = 20 * np.log10(mag)
    fft_chunks.append(db)

fft_chunks = np.array(fft_chunks)

# ==============================================================================
# Set up audio playback with callback
# ==============================================================================
playback_index = 0
def audio_callback(outdata, frames, _timeinfo, _status):
    global playback_index
    end_idx = playback_index + frames
    if end_idx > audio_len:
        chunk = np.zeros((frames,), dtype=audio.dtype)
        remaining = audio_len - playback_index
        if remaining > 0:
            chunk[:remaining] = audio[playback_index:audio_len]
    else:
        chunk = audio[playback_index:end_idx]
    outdata[:] = chunk.reshape(-1,1)
    playback_index += frames

stream = sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    blocksize=BUFFER_SIZE,
    callback=audio_callback
)
stream.start()

# ==============================================================================
# Plot the visualizer
# ==============================================================================
plt.ion()
plt.rcParams['toolbar'] = 'none' # Just for aesthetics
plt.style.use('dark_background')
fig, ax = plt.subplots()
line, = ax.plot(freqs, np.zeros_like(freqs))
ax.set_xlabel("Freq (Hz)")
ax.set_ylabel("dB")
ax.set_xscale("log")
ax.set_xlim(20, 20000)
ax.set_ylim(-40, 60)

while playback_index < audio_len:
    chunk_idx = playback_index // BUFFER_SIZE
    if chunk_idx >= len(fft_chunks):
        break

    line.set_ydata(fft_chunks[chunk_idx])
    fig.canvas.draw_idle()
    plt.pause(0.001)

plt.ioff()
plt.show()
stream.stop()
stream.close()