import wave
import struct
import sys
from Golomb import Golomb
from tqdm import tqdm
from bitstring import BitArray
import math

def diff_encode(fname,golomb_m=512):
    sound = wave.open(fname, "r")

    buffer = BitArray()

    # File attributes
    n_channels = sound.getnchannels()
    samp_width = sound.getsampwidth()
    framerate = sound.getframerate()

    prev_left = 0
    prev_right = 0
    outSamples = []
    print("Starting Golomb encoded differences. No probabilities.")
    for _ in tqdm(range(sound.getnframes())):
        sample = sound.readframes(1)
        valLeft, valRight = struct.unpack('<hh', sample)
        
        # Calculating current difference
        d_left = valLeft - prev_left
        d_right = valRight - prev_right

        # Updating last difference
        prev_left = valLeft
        prev_right = valRight

        # 0 if positive, 1 if negative
        if d_left > -1:
            valLeft = '0'+Golomb.to_golomb(d_left,golomb_m)
        else:
            valLeft = '1'+Golomb.to_golomb(d_left,golomb_m)

        if d_right > -1:
            valRight = '0'+Golomb.to_golomb(d_right,golomb_m)
        else:
            valRight = '1'+Golomb.to_golomb(d_right,golomb_m)

        #print(valLeft,valRight)
        buffer += BitArray(bin=valLeft) + BitArray(bin=valRight)

        #outSamples.append(struct.pack('<xx', valLeft, valRight))

    sound.close()

    print("Done. Writing file...")
    # Write new file
    out = wave.open(fname+"_encoded_m{}.wav".format(golomb_m), "w")
    out.setnchannels(n_channels)
    out.setsampwidth(samp_width)
    out.setframerate(framerate)
    out.writeframesraw(buffer.tobytes())
    out.close()
    print("File written.")

def diff_decode(fname,golomb_m=512):
    sound = wave.open(fname, "rb")

    # File attributes
    n_channels = sound.getnchannels()
    samp_width = sound.getsampwidth()
    framerate = sound.getframerate()

    prev_left = 0
    prev_right = 0

    readingLeft = True
    isNegative = None
    startGolomb = -1
    endGolomb = -1
    startBinary = -1
    endBinary = -1
    c = int(math.ceil(math.log(golomb_m,2)))

    print("Starting Golomb decoding.")
    # We don't know how many "soundframes" there are, so using sys.maxsize, it reads everything it can
    data = BitArray(sound.readframes(sys.maxsize))
    for i,bit in enumerate(data):
        # Sign bit
        if isNegative == None:
            isNegative = bit
        # Code bits
        else:
            if startGolomb < 0:
                startGolomb = i
            if not bit:
                # End of unary
                startBinary = i
                continue
            if startBinary > 0:
                if bit:
                    pass
                    
        
        print(bit)
        print(data[i:i+2])

    #outSamples.append(struct.pack('<xx', valLeft, valRight))

    sound.close()

    print("Done. Writing file...")
    # Write new file
    out = wave.open(fname+"_decoded.wav", "w")
    out.setnchannels(n_channels)
    out.setsampwidth(samp_width)
    out.setframerate(framerate)
    out.writeframesraw(leftBuffer.tobytes())
    out.writeframesraw(rightBuffer.tobytes())
    out.close()
    print("File written.")

# BARELY STARTED. TODO complete function when the base function is done
def diff_encode_dict(fname):
    sound = wave.open(fname, "r")

    # File attributes
    n_channels = sound.getnchannels()
    samp_width = sound.getsampwidth()
    framerate = sound.getframerate()

    prev_left = 0
    prev_right = 0
    diff_left = {}
    diff_right = {}
    for _ in range(sound.getnframes()):
        sample = sound.readframes(1)
        valLeft, valRight = struct.unpack('<hh', sample)
        #sample = struct.pack('<hh', valLeft, valRight)
        
        # Calculating current difference
        d_left = valLeft - prev_left
        d_right = valRight - prev_right

        # Updating last difference
        prev_left = valLeft
        prev_right = valRight

        # Storing counts in dictionary
        if d_left in diff_left:
            diff_left[d_left] += 1
        else:
            diff_left[d_left] = 1

        if d_right in diff_right:
            diff_right[d_right] += 1
        else:
            diff_right[d_right] = 1
    sound.close()

    differencesLeft = [(k, diff_left[k]) for k in sorted(diff_left, key=diff_left.get, reverse=True)]
    differencesRight = [(k, diff_right[k]) for k in sorted(diff_right, key=diff_right.get, reverse=True)]


    # Write new file
    copy = wave.open("encoded.wav", "w")
    copy.setnchannels(n_channels)
    copy.setsampwidth(samp_width)
    copy.setframerate(framerate)
    for sample in samples:
        copy.writeframesraw(sample)
    copy.close()

def main():
    if len(sys.argv) < 2:
        print("USAGE: python3 {} sound_file.wav".format(sys.argv[0]))
        return
    fname = sys.argv[1]

    diff_encode(fname,1024)
    #diff_decode(fname)

if __name__ == "__main__":
    main()
