# GPTCRE/tools.py

import matplotlib.pyplot as plt

def plot_pitch(pitch, sr, hop_length):
    times = np.arange(len(pitch)) * hop_length / sr
    plt.figure(figsize=(10, 4))
    plt.plot(times, pitch, label='Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (MIDI number)')
    plt.title('Pitch Tracking')
    plt.legend()
    plt.show()
