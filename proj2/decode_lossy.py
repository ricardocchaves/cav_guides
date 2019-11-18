import sys
import wave
import struct
import sys
from Golomb import Golomb
from tqdm import tqdm
from bitstring import BitArray
import math

numberOfBitsOriginal = 16
numberOfBitsNew = 7

### Golomb decoding function
### Opens the file fname, gets the headers required for the decoding process and writes the
### decoded WAVE file to outFname
def decode_lossy(fname, outFname):
    sound = wave.open(fname, "rb")

    # File attributes
    n_channels = sound.getnchannels()
    samp_width = sound.getsampwidth()
    framerate = sound.getframerate()

    outSamples = BitArray()

    print("Starting Golomb decoding.")
    # Get header information
    from pickle import loads
    header_size = struct.unpack('<i',sound.readframes(1))[0]
    header_serialized = sound.readframes(math.ceil(header_size/4))
    header = loads(header_serialized)
    left_m = header['m_left']
    right_m = header['m_right']
    left_symbolToValue = header['left_symbolToValue']
    right_symbolToValue = header['right_symbolToValue']

    # We don't know how many "soundframes" there are, so using sys.maxsize, it reads everything it can
    data = BitArray(sound.readframes(sys.maxsize)).bin

    # Golomb attributes for left channel
    left_log_m = math.log(left_m,2)
    left_c = int(math.ceil(left_log_m))
    left_div = int(math.pow(2,left_c) - left_m)

    # Golomb attributes for right channel
    right_log_m = math.log(right_m,2)
    right_c = int(math.ceil(right_log_m))
    right_div = int(math.pow(2,right_c) - right_m)

    i = 0
    samplesDecoded = 0
    signValue = 1
    readingLeft = True
    valLeft = None
    valRight = None
    prevLeft = 0
    prevRight = 0
    pbar = tqdm(total=len(data))
    while i < len(data):
        if readingLeft:
            c = left_c
            div = left_div
            golomb_m = left_m
        else:
            c = right_c
            div = right_div
            golomb_m = right_m

        ### Start of Golomb coding
        # Unary coding
        pos = 0
        if i>=len(data):
            break
        while data[i] == '1':
            i += 1
            pos += 1
            pbar.update(1)
        q = pos  #q -> '0'

        i +=1
        pbar.update(1)

        # Binary remainder
        if i+c-1>=len(data):
            break
        if int(data[i:i+c-1],2)>=div:
            #ler c, skip c-1
            if i+c>=len(data):
                break
            # If remainder is of length c, because of only c-1 '1' bits
            r = int(data[i:i+c],2)
            i += c
            pbar.update(c)
        else:
            #ler c-1, skip c-1-1
            r = int(data[i:i+c-1],2)
            i += c-1
            pbar.update(c-1)

        if r >= div:
            r = r - div
        val = (q*golomb_m + r)*signValue # decoded value from Golomb

        if readingLeft:
            d_left = int( left_symbolToValue[val] * math.pow(2,numberOfBitsOriginal - numberOfBitsNew) )
            valLeft = d_left+prevLeft
            prevLeft = valLeft
            readingLeft = False
        else:
            d_right = int( right_symbolToValue[val] * math.pow(2,numberOfBitsOriginal - numberOfBitsNew) )
            valRight = d_right+prevRight
            prevRight = valRight
            readingLeft = True

        if valRight != None and valLeft != None:
            samplesDecoded += 1
            outSamples += struct.pack('<hh', valLeft, valRight)
            valRight = None
            valLeft = None
    pbar.close()

    # Write new file
    print("Done. Writing file...")
    out = wave.open(outFname, "w")
    out.setnchannels(n_channels)
    out.setsampwidth(samp_width)
    out.setframerate(framerate)
    outBytes = outSamples.tobytes()
    print("Writing {} bytes of data.".format(len(outBytes)))
    out.writeframesraw(outBytes)
    out.close()
    print("File written: {}".format(outFname))

def main():
    if len(sys.argv) < 2:
        print("USAGE: python3 {} encoded_sound_file.wav".format(sys.argv[0]))
        return
    fname = sys.argv[1]
    decode_lossy(fname,"./decoded_file.wav")

if __name__ == "__main__":
    main()
