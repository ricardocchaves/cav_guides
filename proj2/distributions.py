import wave
import struct
import sys
from Golomb import Golomb
from tqdm import tqdm
from bitstring import BitArray
import math

import matplotlib.pyplot as plt
import numpy as np
import operator

from scipy.optimize import curve_fit


def readWavFile(fname):
    sound = wave.open(fname, "r")

    buffer = BitArray()

    # File attributes
    n_channels = sound.getnchannels()
    samp_width = sound.getsampwidth()
    framerate = sound.getframerate()

    prev_left = 0
    prev_right = 0
    sample_values = []
    diff_values = []
    
    for _ in tqdm(range(sound.getnframes())):
        sample = sound.readframes(1)
        valLeft, valRight = struct.unpack('<hh', sample)
    
        sample_values += [(valLeft,valRight)]

        d_left = valLeft - prev_left
        d_right = valRight - prev_right

        # Updating last difference
        prev_left = valLeft
        prev_right = valRight

        diff_values += [(d_left,d_right)]

    return sample_values, diff_values


def prob(data):

    d = {}
    samples = 0
    for value in data:
        if value in d:
            v = d.get(value)
            d.update({value: v+1 })
        else:
            d.update({value: 1})
        samples += 1
    for k in d:
        d.update({k: d.get(k)/samples})
    
    return d
    #d = prob de cada sample, tranformed = prob de cada simbolo, mapping = simbolo de cada sample
    #mapeament está "aleatório"

def map_symbols( sample_probability ):

    sorted_samples = sorted(sample_probability.items(), key=operator.itemgetter(1), reverse=True)
    #sample from most likely to least

    transformed = {}
    symbol_mapping = {}
    i = 0
    for k in sorted_samples:
        symbol_mapping.update({i: k})
        i += 1

    return symbol_mapping

def findBestGolomb(symbol_mapping):
    x = [k for k in symbol_mapping.keys()]
    y = [symbol_mapping.get(k)[1] for k in symbol_mapping.keys()]

    popt, pcov = curve_fit(golomb_pattern, x, y)

    print(popt)

    plt.figure(figsize=(7, 5))
    plt.plot( x, golomb_pattern(x, *popt), 'bo', markersize=0.5)
    plt.ylabel('curve')
    plt.rcParams['agg.path.chunksize'] = 10000
    #plt.axis([1,len(samples)+1, -2**15, 2**15])
    plt.show(block = False)

    plt.plot( x, y, 'ro', markersize=0.5)
    plt.ylabel('probability')
    plt.rcParams['agg.path.chunksize'] = 10000
    #plt.axis([1,len(samples)+1, -2**15, 2**15])
    plt.show(block = True)

    alpha = popt[0]
    m_param = math.ceil( -1/math.log2(alpha) )

    return alpha, m_param

def golomb_pattern(x, alpha):
    return (alpha**x)*(1-alpha)

def main():
    if len(sys.argv) < 2:
        print("USAGE: python3 {} sound_file.wav".format(sys.argv[0]))
        return
    fname = sys.argv[1]

    samples, diff_samples = readWavFile(fname)
    #print(samples)

    sample_probability = prob([r for (l,r) in diff_samples])

    symbol_mapping = map_symbols(sample_probability)
    #{ symbol: (value, probability)}

    alpha, m_param = findBestGolomb(symbol_mapping)
    print(alpha,m_param)

    #probability of each sample value
    plt.figure(figsize=(7, 5))
    plt.plot( [k for k in sample_probability.keys()], [sample_probability.get(k) for k in sample_probability.keys()], 'ro', markersize=0.5)
    plt.ylabel('probability')
    plt.rcParams['agg.path.chunksize'] = 10000
    #plt.axis([1,len(samples)+1, -2**15, 2**15])
    plt.show(block = False)

    #probability of each symbol (choosen by probability)
    # plt.figure(figsize=(7, 5))
    # plt.plot( [s for s in symbol_mapping.keys()], [symbol_mapping.get(s)[1] for s in symbol_mapping.keys()], 'ro', markersize=0.5)
    # plt.ylabel('probability')
    # plt.rcParams['agg.path.chunksize'] = 10000
    # #plt.axis([1,len(samples)+1, -2**15, 2**15])
    # plt.show(block = True)

    # #Golomb pattern
    # plt.figure(figsize=(7, 5))
    # plt.plot( [x for x in range(0, 100)], [golomb_pattern(x,0.5) for x in range(0,100)], 'ro', markersize=0.5)
    # plt.ylabel('probability')
    # plt.rcParams['agg.path.chunksize'] = 10000
    # #plt.axis([1,len(samples)+1, -2**15, 2**15])
    # plt.show(block = True)

    
    '''
    plt.figure(figsize=(10, 10))
    plt.plot( [ s+1 for s in range(0,len(samples))], [r for (l,r) in samples])
    plt.ylabel('some numbers')
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.axis([1,len(samples)+1, -2**15, 2**15])
    plt.show(block = False)

    plt.figure(figsize=(10, 10))
    x = [r for (l,r) in diff_samples]
    plt.hist(x, density=True)
    plt.ylabel('Probability')
    plt.show(block = False)
    
    plt.figure(figsize=(10, 10))
    plt.plot( [ s+1 for s in range(0,len(diff_samples))], [r for (l,r) in diff_samples])
    plt.rcParams['agg.path.chunksize'] = 10000
    #plt.axis([1,len(samples)+1, -2**15, 2**15])
    plt.show()
    '''
    


if __name__ == "__main__":
    main()

