# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:01:43 2023

@author: waszee
"""

import torch
import torchaudio
import matplotlib.pyplot as plt



def plot_waveform(waveform, sample_rate, title="Waveform"):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    
    
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate, cmap="twilight")
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
  

def main():
    torchaudio.set_audio_backend("soundfile")
    print(torch.__version__)
    print(torchaudio.__version__)
    SAMPLE_PATH = "7011.wav"
    metadata = torchaudio.info(SAMPLE_PATH)
    print(metadata)
    waveform, sample_rate = torchaudio.load(SAMPLE_PATH)
    num_frames, num_channels = waveform.shape
    print(num_frames, num_channels, sample_rate)
    plt.show(block=False)
    plot_waveform(waveform, sample_rate)
    plot_specgram(waveform, sample_rate)
    
    
   

if __name__ == "__main__":
    main()

