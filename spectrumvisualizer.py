import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

plt.ion()
fig, ax = plt.subplots()

# Initial empty line
line, = ax.plot([], [])
ax.set_xlabel("Freq (Hz)")
ax.set_ylabel("dB")
ax.set_xscale("log")
ax.set_xlim(20, 20000)
ax.set_ylim(0, 140)

sampleRate, stereoAudio = wavfile.read("key_whatsgood.wav")
audio = stereoAudio[:, 0]

# Get buffer
bufferSize = 1024 # Simulate a 1024 sample buffer from the audio file
window = np.hanning(bufferSize)

# Plot audio to the spectrum visualizer
for i in range(0, len(audio), bufferSize):
    buffer = audio[i:bufferSize+i] * window
    # Apply FFT
    fft = np.fft.rfft(buffer)

    # Get magnitude for mapping
    mag = np.abs(fft)

    # Map FFT to freq log grid
    freqs = np.fft.rfftfreq(bufferSize, 1/sampleRate)

    # Update line data
    line.set_xdata(freqs)
    line.set_ydata(20*np.log10(mag))
    
    ax.relim()
    ax.autoscale_view()
    
    plt.draw()
    plt.pause(0.001)