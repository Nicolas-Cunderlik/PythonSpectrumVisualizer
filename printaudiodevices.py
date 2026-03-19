# Use this to print the audio devices on your system, and put them in the bettervisualizer.py

import sounddevice as sd

devices = sd.query_devices()
print(devices)