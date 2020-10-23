# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 18:31:57 2020 to save CW as Numpy array to


PyTorch Tensor; individual code characters, words, or messages okay
The Coded is MIT license  with no warranty implied. Use at own risk.
@author: waszee
users need to have /TensorData folder below where this is run or edit below
messages should be short in this version as used for filename of tensor
"""

# import sounddevice as sd
import pygame
import sys
import numpy as np
# import time
# from time import sleep
import matplotlib.pyplot as plt
import torch as pyt
import soundfile as sf

CODE = {'A': '.-', 'B': '-...', 'C': '-.-.',
        'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..',
        'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---',
        'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-',
        'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..',

        '0': '-----', '1': '.----', '2': '..---',
        '3': '...--', '4': '....-', '5': '.....',
        '6': '-....', '7': '--...', '8': '---..',
        '9': '----.',

        ',': '--..--', '.': '.-.-.-', '?': '..--..',
        '/': '-..-.', '-': '-....-', '!': '-.-.--',
        '(': '-.--.', ')': '-.--.-', '&': '.-....',
        ':': '---...', ';': '-.-.-.', '=': '-...-',
        '+': '.-.-.', '_': '..--.-', '"': '.-..-.',
        '$': '...-..-', '@': '.--.-.',

        '<': '.-...', '>': '.-.-.',
        '%': '...-.-',
        '{': '....--', '}': '...-.',

        "'": '.----.',

        '\n': '-...-.-',  # line feed = <BK>  same as \x0a

        '\u00c4': '.-.-',  # a umlat
        '\u00d6': '---.',  # o umlat
        '\u00dc': '..--',  # u umlat
        '\u03b2': '...--..',  # beta

        }
LRANGE = 200
URANGE = 12000
def verify(string):
    keys = CODE.keys()
    for char in string:
        if char.upper() not in keys and char != ' ':
            sys.exit('Error - a character ' + char + ' in the msg cannot be translated to Morse Code')


def beat(tlen, samrate, tone):
    t = np.arange(0, tlen, 1 / samrate)
    asnd = 0.4 * np.sin(2 * np.pi * tone * t)  # was arbitrary based on comment of a previous morse coder
    # add a ramp up and down to reduce sound clipping
    nmax = np.size(asnd)
    nten = 100
    pi2 = np.pi * 2
    for n in range(0, nten):
        x = np.sin(pi2 / (nten - n))
        asnd[n] = asnd[n] * x
        if asnd[n] > 0.4: asnd[n] = 0.4
        asnd[nmax - 1 - n] = asnd[nmax - 1 - n] * x * 4
        if asnd[nmax - 1 - n] > 0.4: asnd[nmax - 1 - n] = 0.4

    bsnd = asnd.clip(min=-0.4, max=0.4)
    # bsnd  = (bsnd * 32768).astype(np.int16) # will do this for snd out later
    # bsnd = (bsnd * 32768).astype(np.float)  #PyTorch likes to use floats for data
    return bsnd

    # asnd  = (asnd * 32768).astype(np.int16)
    # return asnd


def adjchunk(c):
    CHUNK = 1024
    z1 = np.size(c)
    z2 = z1 / CHUNK  # assuming chunk size is 1024
    # print('ratio sampleframes/chunk =', z2)
    # plt.plot(c)
    # plt.show()
    z3 = int(z2) * CHUNK
    z4 = z3 - z1
    # print('diff from whole chunk =', z4)
    # ketters are padded with 4 spaces to begin so slice from front
    d = c[-z4:]
    # print('size of d =', np.size(d), ' ratio is ',np.size(d)/CHUNK)
    d = (d * 32768).astype(np.int16)
    return d


def tensorsave(msg, cw_msg):
    try:
        print(msg)
        filename = msg + '.t'
        print(filename)
        cwTen = pyt.from_numpy(cw_msg)
        pyt.save(cwTen, './TensorData/' + filename)


    except Exception as e:
        print('Failed trying to save as tensor array to file: ' and e)
        plt.close()
        sys.exit()


def SaveSnd(filename, data, fs=44100, channels=1):
    try:
        if filename == "":
            return
        else:
            filename = './SoundWave/' + filename + ".wav"
            with sf.SoundFile(filename, 'w', fs, channels) as f:
                f.write(data)
    except Exception as e:
        print("exception:", e)

def OpenSnd(filename,fs=44100, channels=1):
    try:
        if filename == "":
            return
        else:
            filename = './SoundWave/' + filename + ".wav"
            print("Opening file: ", filename)
            data, samplerate = sf.read(filename)
            # print(data.shape, " sr:", samplerate)
            return data, samplerate
    except Exception as e:
        print("exception:", e)

def create_spectrogram(ts, NFFT, sample_rate, noverlap=None):
    '''
    ts: original time series
    NFFT: The number of data points used in each block for the DFT.
    Fs: the number of points sampled per second, so called sample_rate
    noverlap: The number of points of overlap between blocks. The default
    value is NFFT/2.
    '''
    if noverlap is None:
        noverlap = NFFT / 2
    noverlap = int(noverlap)

    starts = np.arange(0, len(ts), NFFT - noverlap, dtype=int)
    # remove any window with less than NFFT sample size
    starts = starts[starts + NFFT < len(ts)]
    xns = []
    f = []
    for start in starts:
        f, pxx = get_spectrum(ts[start:start + NFFT], sample_rate)
        xns.append(pxx)
    spec = np.array(xns).T
    # plt.plot(f, spec)
    # plt.show()
    assert spec.shape[1] == len(starts)
    return starts, spec


def plot_spectrogram(msg, cw_msg, spec, sample_rate, L, starts, mappable=None):
    fig, [ax1, ax2] = plt.subplots(2, 1, constrained_layout=True)
    cwlen = len(cw_msg)
    secs = cwlen/sample_rate
    fig.suptitle(msg + "\n audio length: {}, or seconds: {},  Spec.shape={}".format(cwlen, secs, spec.shape))
    ax1.plot(cw_msg)
    ax1.set_ylabel('amp')
    ax1.set_xlabel('Audio Sample (sample rate*seconds)')
    ax2.set_ylabel('freq(Hz)')
    ax2.set_xlabel("seconds")

    #the audio bandwidth depends on your receiver capabilies, adjust urange and lrange to set window
    vmax = spec.shape[0]
    rlow = int(2 * vmax * LRANGE / sample_rate)
    rhigh = int(2 * vmax * URANGE / sample_rate)
    specshow = spec[rlow:rhigh, ]
    #added some padding to keep emissions from being too narrow to see and help plot view
    specshowx = np.repeat(specshow, axis=0, repeats=2)
    # rescale the data along time axis to -1 to 1 range
    dscaled = scale(specshowx, out_range=(-1, 1), axis=None)

    # xstretch is just a factor to help view differ amounts of signals - not optimized yet
    xstretch = secs*1000/(URANGE-LRANGE)
    im = ax2.imshow(dscaled, origin='lower', cmap=plt.get_cmap('hot'), aspect=xstretch)

    Nyticks = 5
    fig.canvas.draw()
    labels = ax2.get_yticklabels()
    yticks = ax2.get_yticks()
    v = specshowx.shape[0]
    ks = np.linspace(0, v, Nyticks)
    ax2.set_yticks(ks)
    ksHz = np.linspace(LRANGE, URANGE, Nyticks)
    ax2.set_yticklabels(ksHz)
    ax2.set_ylabel("Frequency (Hz)")
    #
    Nxticks = 10
    timeticks = np.linspace(0, specshowx.shape[1], Nxticks)
    ax2.set_xticks(timeticks)
    # timelabels = np.linspace(0, secs, Nxticks)
    timelabels = ["{:4.2f}".format(i) for i in np.linspace(0, secs, Nxticks)]
    ax2.set_xticks(timeticks)
    ax2.set_xticklabels(timelabels)
    ax2.set_xlabel("Time (sec)")

def get_spectrum(data, sample_rate):
    timestep = 1.0 / sample_rate
    n = data.shape[0]
    # print("n:", n)
    f = np.fft.fftfreq(n, timestep)
    # f = np.fft.fftshift(f)
    w = np.blackman(n)
    pxx = np.fft.fft(data * w)
    # pxx = np.fft.fftshift(pxx)
    # pxx = 2 / n * np.abs(pxx)
    pxx = pxx.real**2 + pxx.imag**2
    # print(f.shape, np.max(f), np.min(f))
    freal = f[0:int(n/2)]
    preal = pxx[0:int(n/2)]
    # print(freal.shape, " ", pxx.shape)
    return freal, preal

def scale(x, out_range=(-1, 1), axis=None):
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def main():
    print('Welcome to Python Code Out ver 0.2\n')
    print('Type exit to exit message loop or ^C to abort')
    print('Default 17 wpm, 750 Hz tone, 44100 sample rate, 1 channel mono -- type setup to change')

    samrate = 44100  # audio sampling rate
    T = 0.06    # .08=10wpm, 0.07=14wpm. 0.06=17wpm, 0.05=20 wpm,
    # T3 = 3*T  #
    # farn=1.0    a modifier like Farnsworth  to lengthen space between sounds or letters
    tone = 750  # audio tone freq
    # sd.default.samplerate = samrate
    # sd.default.channels = 1
    # sd.default.dtype = np.int16

    pygame.mixer.pre_init(samrate, size=-16, channels=1)
    pygame.mixer.init()
    pygame.init()
    cw_msg = [], []
    # createHi  dit, day and intraspace as numpy array values

    try:
        msg = input('Enter exit, setup, read, or create: ')
        # handle special messages and check others to be sure will be handled
        if msg == 'exit':
            sys.exit()
        if msg == 'setup':
            msg = input('Enter dit milliseconds = ')
            T = float(msg) / 1000
            # T3=3*T
            msg = input('Tone in Hz = ')
            tone = int(msg)
            tone = int(tone)
            msg = input('Audio Sample Rate (counts/sec) = : ')
            samrate = int(msg)
            msg = ""
        if msg == 'read':
            filename = input("Enter filename:")
            cw_msg, samrate = OpenSnd(filename)
        if msg == 'create':
            dit = beat(T, samrate, tone)
            dah = beat(T * 3, samrate, tone)
            # p2.plot(dah)
            space = np.zeros(dit.size, np.float)
            wspace = np.zeros(dit.size * 5, np.float)  # 2spaces + wspace gives 7 spaces
            # fig = plt.figure()
            # p1 = fig.subplots(1)
            # p1.plot(dit)
            # plt.show()
            while True:
                msg = input('Enter Message, exit, or [,] special keys: ')
                cw_msg = space
                if msg == '[' or ']':
                    if msg == '[':
                        msg = 'CQ CQ CQ DE MYCALL  K'
                    if msg == ']':
                        msg = 'DE MYCALL KN'
                else:
                    verify(msg)
                print()

                if msg == 'exit':
                    break
                c = space
                for char in msg:
                    if char == ' ':
                        print()
                        c = np.concatenate((c, space, wspace))
                    else:
                        print(char + ' ' + CODE[char.upper()])
                        c = space
                        for char in CODE[char.upper()]:

                            if char == '.':
                                b = dit
                            else:
                                b = dah
                            c = np.concatenate((c, space, b))

                        c = np.concatenate((c, space, space))
                        d = adjchunk(c)
                        dsnd = pygame.sndarray.make_sound(d)
                        dsnd.play()
                        # sd.play(d)
                        count = 0
                        while pygame.mixer.get_busy():
                            count = 1 + count

                    # pygame.time.wait(int(T3*1000)) #ketter spacing interval
                    cw_msg = np.concatenate((cw_msg, c))
                    c = space


                # plt.plot(cw_msg)
                # plt.show()
                # print()
                # the message, msg, is used as the filename, stored as .wav and .t files
                SaveSnd(msg, cw_msg)
                tensorsave(msg, cw_msg)
                # save = "n"
                # # save = input("save data? (y/n):")
                # if save == "y":
                #     with open('test.npy', 'wb') as f:
                #         np.save(f, ts)


        # now show freq domain plot of the message or loaded wavefile
        L = 1024
        noverlap = 512
        sample_rate = samrate
        starts, spec = create_spectrogram(cw_msg, L, sample_rate, noverlap=noverlap)
        plot_spectrogram(msg, cw_msg, spec, sample_rate, L, starts)
        plt.show()


    except KeyboardInterrupt:
        print('interupt by keyboard control c detected')

    except Exception as e:
        print('Exception: ' and e)
        plt.close()
        sys.exit()

    print('\n\nGoodbye!')
    plt.close()


if __name__ == "__main__":
    main()
