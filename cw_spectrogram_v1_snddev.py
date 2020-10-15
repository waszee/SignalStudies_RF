#!/usr/bin/python

"""
based on
PyAudio + PyQtGraph Spectrum Analyzer
Author:@sbarratt
Date Created: August 8, 2015
and Spectrum Analyzer with STFT see Yumi's blog
https://fairyonice.github.io/implement-the-spectrogram-from-scratch-in-python.html
as modified by waszee Oct 12, 2020
this version is using sounddevice instead of pyaudio 
"""
import argparse
#import pyaudio
import struct
#import math
import sys
import numpy as np
import IPython as ipy
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import matplotlib.pyplot as plt
import queue
import sounddevice as sd
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-d', '--device', type=int,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
args = parser.parse_args(remaining)

q = queue.Queue()

# Audio Format (check Audio MIDI Setup if on Mac)
FORMAT = np.int16
RATE = 44100
CHANNELS = 1

# Set Plot Range [-RANGE,RANGE], default is nyquist/2
URANGE =12000 #used oyqtgraph not in matlabplot at the moment
if not URANGE:
    URANGE = RATE/2
LRANGE=200
if not LRANGE:
    LRANGE=0   
TRACK=1024

OVERLAP=400
COLLECTSEC=30
#expect 441000 data points per sec so this about megabyte of data
# input block is used for the realtime pyqtgraph
INPUT_BLOCK_TIME = 0.1
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

#print("block:",INPUT_FRAMES_PER_BLOCK)
# Which Channel if stereo? (L or R)
LR = "l"

class SpectrumAnalyzer():
    def __init__(self):
        #self.pa = pyaudio.PyAudio()
        #self.initMicrophone()
        #self.sd=sounddevice.device
        self.sdinit_stream()
        
        
        self.initUI()

    def find_input_device(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            if devinfo["name"].lower() in ["mic","input"]:
                device_index = i
        return device_index

    def initMicrophone(self):
        device_index = self.find_input_device()
        self.stream = self.pa.open(    format = FORMAT,
                                    channels = CHANNELS,
                                    rate = RATE,
                                    input = True,
                                    input_device_index = device_index,
                                    frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

    def sdinit_stream(self):
        self.sdstream = sd.InputStream(samplerate=args.samplerate, device=args.device,
                            channels=args.channels,blocksize =INPUT_FRAMES_PER_BLOCK,callback = self.sdcallback, dtype=np.int16)
        self.sdstream.start()
    
    def sdcallback(self,indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status)
        ndata=indata[:]  
        q.put(ndata)
        
    def sdreadData(self):
       
        try:
            data = q.get()
        except queue.Empty:
            pass
        return data
#         block=[]
#         while (len(block) < INPUT_FRAMES_PER_BLOCK):
#             block += self.sdstream.read(INPUT_FRAMES_PER_BLOCK)
#             print(len(block))
#         count = len(block)/2
#         format = "%dh"%(count)
#         shorts = struct.unpack( format, block )
# 
#         if CHANNELS == 1:
#             return np.array(shorts)
#         else:
#             l = shorts[::2]
#             r = shorts[1::2]
#             if LR == 'l':
#                 return np.array(l)
#             else:
#                 return np.array(r)

        
#     def readData(self):
#         block = self.stream.read(INPUT_FRAMES_PER_BLOCK)
#         count = len(block)/2
#         format = "%dh"%(count)
#         shorts = struct.unpack( format, block )
# 
#         if CHANNELS == 1:
#             return np.array(shorts)
#         else:
#             l = shorts[::2]
#             r = shorts[1::2]
#             if LR == 'l':
#                 return np.array(l)
#             else:
#                 return np.array(r)

    def initUI(self):
        self.app = QtGui.QApplication([])
        self.app.quitOnLastWindowClosed()
        self.mainWindow = QtGui.QMainWindow()
        self.mainWindow.setWindowTitle("Spectrum Analyzer")
        self.mainWindow.resize(800,300)
        self.centralWid = QtGui.QWidget()
        self.mainWindow.setCentralWidget(self.centralWid)
        self.lay = QtGui.QVBoxLayout()
        self.centralWid.setLayout(self.lay)
        self.specWid = pg.PlotWidget(name="spectrum")
        self.specItem = self.specWid.getPlotItem()
        self.specItem.setMouseEnabled(y=False)
        self.specItem.setYRange(0,5000)
        self.specItem.setXRange(-URANGE,URANGE, padding=0)
        self.specAxis = self.specItem.getAxis("bottom")
        self.specAxis.setLabel("Frequency [Hz]")
        self.lay.addWidget(self.specWid)
        self.mainWindow.show()
        self.app.aboutToQuit.connect(self.close)

 
    def close(self):
        #self.stream.close()
        self.stream.stop()
        sys.exit()

    def get_spectrum(self, data):
        T = 1.0/RATE
        N = data.shape[0]
        f = np.fft.fftfreq(N,T)
        f = np.fft.fftshift(f)
        w = np.blackman(N)
        Pxx = np.fft.fft(data*w)
        Pxx = np.fft.fftshift(Pxx)
        Pxx = 2/N*np.abs(Pxx)
        return f, Pxx

       #return f.tolist(), (np.absolute(Pxx)).tolist()

#     def get_xn(self,Xs,n):
# 
#         '''
#         calculate the Fourier coefficient X_n of
#         Discrete Fourier Transform (DFT)
#         '''
#         L  = len(Xs)
#         ks = np.arange(0,L,1)
#         xn = np.sum(Xs*np.exp((1j*2*np.pi*ks*n)/L))/L
#         return(xn)
 
#     def get_xns(self,ts):
#         '''
#         Compute Fourier coefficients only up to the Nyquest Limit Xn, n=1,...,L/2
#         and multiply the absolute value of the Fourier coefficients by 2,
#         to account for the symetry of the Fourier coefficients above the Nyquest Limit.
# 
#         '''
#         mag = []
#         L = len(ts)
#         for n in range(int(L/2)): # Nyquest Limit
#             mag.append(np.abs(self.get_xn(ts,n))*2)
#         return(mag)

   

    def create_spectrogram(self,ts,NFFT,noverlap = None):
        '''
        ts: original time series
        NFFT: The number of data points used in each block for the DFT.
        Fs: the number of points sampled per second, so called sample_rate
        noverlap: The number of points of overlap between blocks. The default
        value is NFFT/2.
        '''
        if noverlap is None:
            noverlap = NFFT/2
        noverlap = int(noverlap)
        starts  = np.arange(0,len(ts),NFFT-noverlap,dtype=int)
        # remove any window with less than NFFT sample size
        starts  = starts[starts + NFFT < len(ts)]
        xns = []
        for start in starts:
            # short term discrete fourier transform
            #ts_window = get_xns(ts[start:start + NFFT])
            f, Pxx = self.get_spectrum(ts[start:start + NFFT])
            #xns.append(ts_window)
            #stack the new readings in upper half array and transpose to horizontal
            N=len(Pxx) 
            Pxx = 2/N*np.abs(Pxx[N//2:N-1])
            xns.append(Pxx)
        specX = np.array(xns).T
        # rescale the absolute value of the spectrogram as rescaling is standard
        spec = 20*np.log10(specX)
        assert spec.shape[1] == len(starts)
        return(starts,spec)
 
    def plot_spectrogram(self,spec,ks,sample_rate, L, starts,tslen, mappable = None):
        plt.figure(figsize=(7.5,3))
        rlow = int(L*LRANGE/sample_rate)
        rhigh = int(L*URANGE/sample_rate)
        specshow = spec[rlow:rhigh,]
                
        plt_spec = plt.imshow(specshow,origin='lower', cmap="twilight_r")
        ## create ylim
        Nyticks = 10
        V=int(specshow.shape[0])
        ks      = np.linspace(0,V,Nyticks)
        ksHz    = self.get_Hz_scale_vec(ks,sample_rate,V*2)
        plt.yticks(ks,ksHz)
        plt.ylabel("Frequency (KHz)")
 
        ## create xlim
        Nxticks = 10
        ts_spec = np.linspace(0,spec.shape[1],Nxticks)
        total_ts_sec=int(tslen/RATE)
        ts_spec_sec  = ["{:4.2f}".format(i) for i in np.linspace(0,total_ts_sec*starts[-1]/tslen,Nxticks)]
        plt.xticks(ts_spec,ts_spec_sec)
        plt.xlabel("Time (sec)")
        plt.title("Spectrogram L={} Spectrogram.shape={}".format(L,spec.shape))
        #plt.colorbar(mappable,use_gridspec=True)
        plt.show()
        return(plt_spec)

    def get_Hz_scale_vec(self,ks,sample_rate,Npoints):
        maxrange=sample_rate/2
        freq_Hz = ks*sample_rate/Npoints*(URANGE-LRANGE)/maxrange+LRANGE
        freq_Hz  = [int(i) for i in freq_Hz ]
        return(freq_Hz )
   

    def alt_spectrogram(self,ts,sample_rate):
        dt = 1/sample_rate
        t = np.arange(0.0, COLLECTSEC, dt)
        NFFT = TRACK  # the length of the windowing segments
        Fs = int(1.0 / dt)  # the sampling frequency
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.plot(t, ts)
        Pxx, freqs, bins, im = ax2.specgram(ts, NFFT=NFFT, Fs=Fs, noverlap=OVERLAP)
        return
   

    def mainLoop(self):
        ts=[]
        
        #with self.stream:
        while 1:
            # Sometimes Input overflowed because of mouse events, ignore this
            
            while (len(ts)<(RATE*COLLECTSEC)):
                try:
                    #data = self.readData()
                    sddata=self.sdreadData()
                    data=sddata.reshape(-1)
                    #print(data)
                except IOError:
                    continue
                try:
                    f, Pxx = self.get_spectrum(data)
                    self.specItem.plot(x=f,y=Pxx, clear=True)
                    QtGui.QApplication.processEvents()
                    ts=np.concatenate((ts,data))
                except Exception as e:
                    print("exception = ",e)
                    print(len(data))
                    print(len(Pxx)," Pxx shape= ",Pxx.shape)
                    print("ts len",len(ts))
                    sa.close
                    break

            L =TRACK
            noverlap = OVERLAP
            Nxlim=10
            sample_rate=RATE
            ks   = np.linspace(0,len(Pxx),Nxlim)
            starts, spec = self.create_spectrogram(ts,L,noverlap = noverlap )
            tslen=len(ts)
            self.plot_spectrogram(spec,ks,sample_rate,L, starts,tslen)
            self.alt_spectrogram(ts,sample_rate)
            sa.close
            

if __name__ == '__main__':
    sa = SpectrumAnalyzer()
    sa.mainLoop()
    

       