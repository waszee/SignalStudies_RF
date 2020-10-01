# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 18:31:57 2020 to save CW as Numpy array to


PyTorch Tensor; individual code characters, words, or messages okay
The Coded is MIT license  with no warranty implied. Use at own risk.
@author: waszee
users need to have /TensorData folder below where this is run or edit below
messages should be short in this version as used for filename of tensor
"""

#import sounddevice as sd
import pygame
import sys
import numpy as np
#import time
#from time import sleep
import matplotlib.pyplot as plt
import torch as pyt



CODE = {'A': '.-',     'B': '-...',   'C': '-.-.',
        'D': '-..',    'E': '.',      'F': '..-.',
        'G': '--.',    'H': '....',   'I': '..',
        'J': '.---',   'K': '-.-',    'L': '.-..',
        'M': '--',     'N': '-.',     'O': '---',
        'P': '.--.',   'Q': '--.-',   'R': '.-.',
        'S': '...',    'T': '-',      'U': '..-',
        'V': '...-',   'W': '.--',    'X': '-..-',
        'Y': '-.--',   'Z': '--..',

        '0': '-----',  '1': '.----',  '2': '..---',
        '3': '...--',  '4': '....-',  '5': '.....',
        '6': '-....',  '7': '--...',  '8': '---..',
        '9': '----.',

        ',':'--..--',  '.':'.-.-.-',  '?':'..--..',
        '/':'-..-.',   '-':'-....-',  '!':'-.-.--',
        '(':'-.--.',   ')':'-.--.-',  '&':'.-....',
        ':':'---...',  ';':'-.-.-.',  '=':'-...-',
        '+':'.-.-.',   '_':'..--.-',  '"':'.-..-.',
        '$':'...-..-', '@':'.--.-.',

        '<':'.-...',  '>':'.-.-.',
        '%':'...-.-',
        '{':'....--',  '}':'...-.',


        "'":'.----.',

        '\n':'-...-.-', #line feed = <BK>  same as \x0a

        '\u00c4':'.-.-', #a umlat
        '\u00d6':'---.', #o umlat
        '\u00dc':'..--', #u umlat
        '\u03b2':'...--..', # beta
       
        }


def verify(string):
    keys = CODE.keys()
    for char in string:
        if char.upper() not in keys and char != ' ':
            sys.exit('Error - a character ' + char + ' in the msg cannot be translated to Morse Code')


def beat(tlen,samrate,tone):
    t = np.arange(0,tlen,1/samrate)
    asnd = 0.4 * np.sin(2*np.pi*tone*t) # was arbitrary based on comment of a previous morse coder 
    #add a ramp up and down to reduce sound clipping
    nmax = np.size(asnd)
    nten = 100
    pi2 = np.pi * 2
    for n in range(0,nten):
        x =  np.sin(pi2/(nten-n)) 
        asnd[n] = asnd[n] * x
        if asnd[n] > 0.4: asnd[n] = 0.4
        asnd[nmax-1-n]=asnd[nmax-1-n] * x * 4
        if asnd[nmax-1-n] > 0.4: asnd[nmax-1-n] = 0.4
     
    bsnd=asnd.clip(min=-0.4,max=0.4)    
    #bsnd  = (bsnd * 32768).astype(np.int16) # will do this for snd out later
    #bsnd = (bsnd * 32768).astype(np.float)  #PyTorch likes to use floats for data
    return bsnd
       
    #asnd  = (asnd * 32768).astype(np.int16)
    #return asnd

def adjchunk(c):
    CHUNK = 1024
    z1=np.size(c)
    z2= z1/CHUNK  # assuming chunk size is 1024
    #print('ratio sampleframes/chunk =', z2)
    #plt.plot(c)
    #plt.show()
    z3=int(z2)*CHUNK
    z4 =z3-z1
    #print('diff from whole chunk =', z4)
    #ketters are padded with 4 spaces to begin so slice from front 
    d = c[-z4:]
    #print('size of d =', np.size(d), ' ratio is ',np.size(d)/CHUNK)
    d  = (d * 32768).astype(np.int16)
    return d
    

def tensorsave(msg,cw_msg):
    try:
        print(msg)
        filename=msg+'.t'  
        print(filename)
        cwTen = pyt.from_numpy(cw_msg)
        pyt.save(cwTen, './TensorData/'+ filename)
       
        
    except Exception as e:
        print('Failed trying to save as tensor array to file: ' and e)
        plt.close()
        sys.exit()  





def main():
    print ('Welcome to Python Code Out ver 0.2\n')
    print('Type exit to exit message loop or ^C to abort')
    print('Default 15 wpm, 750 Hz tone, 44100 sample rate -- type setup to change')

    samrate = 44100 # audio sampling rate
    T = 0.08 # .08=10wpm, 0.07=14wpm. 0.06=17wpm, 0.05=20 wpm,
    #T3 = 3*T #
    #farn=1.0 #  a modifier like Farnsworth  to lengthen space between sounds or letters
    tone = 700 # audio tone freq
    # sd.default.samplerate = samrate
    # sd.default.channels = 1
    # sd.default.dtype = np.int16

    pygame.mixer.pre_init(samrate, size=-16, channels=1)
    pygame.mixer.init()
    pygame.init()

   #createHi  dit, day and intraspace as numpy array values
    
    
    try:
        msg = input('Enter exit, setup or just a return here: ')            
        #handle special messages and check others to be sure will be handled
        if msg == 'exit': sys.exit()      
        if msg == 'setup':
                msg = input('Enter dit milliseconds = ')
                T = float(msg)/1000
                #T3=3*T
                msg = input('Tone in Hz = ' )
                tone = int(msg)
                tone= int(tone)
                msg = input('Audio S''a]mple Rate (counts/sec) = : ')
                samrate = int(msg)
                msg=""
        
                
        dit = beat(T,samrate,tone)
        dah = beat(T*3,samrate,tone)
        #p2.plot(dah)
        space = np.zeros(dit.size,np.float)
        wspace=np.zeros(dit.size*5,np.float) #2spaces + wspace gives 7 spaces
        fig = plt.figure()
        p1 = fig.subplots(1)
        p1.plot(dit)
        plt.show()
        while True:
            msg = input('Enter Message, exit, or [,] special keys: ')
            cw_msg=space
            if msg== '[' or ']':
                if msg == '[': msg ='CQ CQ CQ DE MYCALL  K'
                if msg == ']': msg ='DE MYCALL KN'
            else:
                verify(msg)
            print ()

            if msg == 'exit': break
            c=space
            for char in msg:
                if char == ' ':
                    print ()
                    c = np.concatenate((c,space,wspace))
                else:
                    print (char + ' ' + CODE[char.upper()])
                    c=space
                    for char in CODE[char.upper()]:

                        if char == '.':
                            b=dit
                        else:
                            b=dah
                        c = np.concatenate((c,space,b))
                                            
                    c = np.concatenate((c,space,space))
                    d = adjchunk(c)
                    dsnd = pygame.sndarray.make_sound(d)
                    dsnd.play()
                    #sd.play(d)
                    count=0
                    while pygame.mixer.get_busy():
                        count = 1 + count
                
                #pygame.time.wait(int(T3*1000)) #ketter spacing interval
                cw_msg = np.concatenate((cw_msg,c))
                c=space
            plt.plot(cw_msg)
            plt.show()
            print()
            tensorsave(msg,cw_msg)

    except KeyboardInterrupt:
        print('interupt by keyboard control c detected')

    except Exception as e:
        print('Exception: ' and e)
        plt.close()
        sys.exit()

    print ('\n\nGoodbye!')
    plt.close()
    
    
if __name__ == "__main__":
    main()

