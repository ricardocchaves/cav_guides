import wave
import struct
import sys
from Golomb import Golomb
from tqdm import tqdm
from bitstring import BitArray
import math

import numpy as np
import operator

from scipy.optimize import curve_fit

numberOfBitsOriginal = 16
numberOfBitsNew = 7

def readWavFile(fname):
    sound = wave.open(fname, "r")

    prev_left = 0
    prev_right = 0
    sample_values = []
    diff_values = []

    for _ in tqdm(range(sound.getnframes())):
        sample = sound.readframes(1)
        valLeft, valRight = struct.unpack('<hh', sample)

        sample_values += [(valLeft,valRight)]

        r_left = valLeft - prev_left
        r_right = valRight - prev_right

        d_left = int(r_left / math.pow(2,numberOfBitsOriginal - numberOfBitsNew) )
        d_right = int(r_right / math.pow(2,numberOfBitsOriginal - numberOfBitsNew) )


        r_left = d_left * math.pow(2,numberOfBitsOriginal - numberOfBitsNew)
        r_right = d_right * math.pow(2,numberOfBitsOriginal - numberOfBitsNew)

        # Updating last difference
        prev_left = r_left + prev_left
        prev_right = r_right + prev_right

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

    valueToSymbol = {}
    symbolToValue = {}
    i = 0
    for k in sorted_samples:
        symbolToValue.update({i: k})
        valueToSymbol.update({k[0]: i})
        i += 1

    return symbolToValue,valueToSymbol

def findBestGolomb(symbol_mapping, show):
    x = [k for k in symbol_mapping.keys()]
    y = [symbol_mapping.get(k)[1] for k in symbol_mapping.keys()]

    popt, pcov = curve_fit(golomb_pattern, x, y)

    if show:
        import matplotlib.pyplot as plt
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
        plt.show(block = False)

    alpha = popt[0]
    m_param = math.ceil( -1/math.log2(alpha) )

    return alpha, m_param

def golomb_pattern(x, alpha):
    return (alpha**x)*(1-alpha)

def main():
    if len(sys.argv) < 3:
        print("USAGE: python3 {} sound_file.wav show".format(sys.argv[0]))
        return
    fname = sys.argv[1]
    show = int(sys.argv[2])

    #1 #Read Samples from file
    samples, diff_samples = readWavFile(fname)
    leftSamples = [l for (l,r) in diff_samples]
    rightSamples = [r for (l,r) in diff_samples]
    #print(samples)


    #2 #Calculate probabilty of each sample value after predictor is applyed
    sample_probability_right = prob([r for (l,r) in diff_samples])
    sample_probability_left = prob([l for (l,r) in diff_samples])

    #show sample values' probability distribution after predictor is applyed
    """
    if(show):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.plot( [k for k in sample_probability.keys()], [sample_probability.get(k) for k in sample_probability.keys()], 'ro', markersize=0.5)
        plt.ylabel('probability')
        plt.rcParams['agg.path.chunksize'] = 10000
        #plt.axis([1,len(samples)+1, -2**15, 2**15])
        plt.show(block = False)
    """

    #3 #Map symbols to sample values accourding to probability
    #{ symbol: (value, probability)}
    left_symbolToValue, left_valueToSymbol  = map_symbols(sample_probability_left)
    right_symbolToValue, right_valueToSymbol = map_symbols(sample_probability_right)

    #show choosen symbols' probability distribution
    """
    if(show):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.plot( [s for s in symbol_mapping.keys()], [symbol_mapping.get(s)[1] for s in symbol_mapping.keys()], 'ro', markersize=0.5)
        plt.ylabel('probability')
        #plt.axis([1,len(samples)+1, -2**15, 2**15])
        plt.show(block = False)
    """

    #4 #Find best Golomb Parameters
    alpha_left, m_left = findBestGolomb(left_symbolToValue,show)
    alpha_right, m_right = findBestGolomb(right_symbolToValue,show)
    print("Found Parameters: \n Alpha: {}, \n M: {} ".format(alpha_left,m_left))
    print("Found Parameters: \n Alpha: {}, \n M: {} ".format(alpha_right,m_right))

    #5 #Encode as Golomb
    #TODO
    symbolsLeft = {}
    symbolsRight = {}
    # Removing probabilities from symbolToValue map. valueToSymbol already doesn't have them.
    for k in left_symbolToValue:
        symbolsLeft.update({k:left_symbolToValue[k][0]})
    for k in right_symbolToValue:
        symbolsRight.update({k:right_symbolToValue[k][0]})

    import codec

    sound = wave.open(fname, "r")
    n_channels = sound.getnchannels()
    samp_width = sound.getsampwidth()
    framerate = sound.getframerate()

    codec.encode(fname+"_encoded",leftSamples,rightSamples,symbolsLeft, left_valueToSymbol,symbolsRight, right_valueToSymbol, m_left, m_right, n_channels, samp_width, framerate)

    if show:
        input("Press enter to exit")

if __name__ == "__main__":
    main()
