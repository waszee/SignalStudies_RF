14054
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:39:42 2023

@author: waszee
code is clipped from the torchaudio tutorials and tweaked for windows
and a microphone attached via a usb dongle device.  Used FFMPEG query to
determine the actual address for direct show device as tutorial suggested
by Moto Hira [torchaudio documention v2.0]
ffmpeg -f dshow -list_devices true -i dummy

"""
import torch
import torchaudio
import datetime
print(torch.__version__)
print(torchaudio.__version__)

try:
    from torchaudio.io import StreamReader
except ModuleNotFoundError:
    try:
        
        print(
            """
           Module not found
            """
        )
    except ModuleNotFoundError:
        pass
    raise

import matplotlib.pyplot as plt

#sample_rate =  16384
sample_rate=8192
now = datetime.datetime.now()
print (now.strftime ("%Y-%b-%d, %A %I:%M:%S"))
sfreq = (input("What freq are you monitoring in Khz?: ") or "14?")


# fmt: off
descs = [
    # No filtering
    "anull",
    # Apply a highpass filter then a lowpass filter
    "highpass=f=800,lowpass=f=1000",
    
]
       
# Micorphone selected using ffmpeg device 
src = "audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{9C528C6B-6FA9-4FCB-90E5-1A8013348462}"
#src = "Microphone (3- USB Audio Device)"
streamer = StreamReader(src,"dshow")   #, buffer_size=CHUNKSIZE)

print("The number of source streams:", streamer.num_src_streams)
print(streamer.get_src_stream_info(0))

for desc in descs:
    streamer.add_audio_stream(
        #frames_per_chunk=163840,
        frames_per_chunk=sample_rate * 12,
        filter_desc=f"aresample={sample_rate},{desc},aformat=sample_fmts=fltp",
    )
    

for i in range(streamer.num_out_streams):
    print(i,streamer.get_out_stream_info(i))
        
fig, axs = plt.subplots(6, 1)
axs[0].grid(True)
axs[0].set_ylim([-1,1])
plt.setp(axs[0].get_xticklabels(), visible=False)
stitle=str(now.strftime ("%Y-%b-%d_ %A %Ih%Mm%Ss ")+ sfreq +"Khz ")

thewaveform,filtwav = [],[]
#axs[0].plot(waveform)
#fig.show()

for i in range(6):
    chunks = next(streamer.stream())
    waveform = chunks[0][:,0]
    axs[0].set_ylim([-1,1])
    axs[0].plot(waveform)
    axs[i].specgram(waveform, Fs=sample_rate, cmap='twilight', detrend='linear' )
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    thewaveform.append(chunks[0])

fig.suptitle("Time in seconds on " +stitle)
sy=(input("save -image?(y/n): ") or "n")
if sy=="y" : plt.savefig(stitle)

#print(thewaveform[0].shape)
#axs[0].plot(thewaveform[0][:,0])
#sw=(input("save -waveform?(y/n): ") or "n")
#if sw=="y": torchaudio.save(stitle+".wav",waveform,sample_rate)

   #display(1)
   #plt.show()



