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
    print("Starting Golomb encoded differences. No probabilities.")
    for _ in tqdm(range(sound.getnframes())):
        sample = sound.readframes(1)
        valLeft, valRight = struct.unpack('<hh', sample)
        
        # Calculating current difference
        d_left = valLeft - prev_left
        d_right = valRight - prev_right

        if _ < 10:
            print("left "+str(d_left))
            print("right "+str(d_right))

        # Updating last difference
        prev_left = valLeft
        prev_right = valRight

        # 0 if positive, 1 if negative
        if d_left >= 0:
            valLeft = '0'+Golomb.to_golomb(d_left,golomb_m)
        else:
            valLeft = '1'+Golomb.to_golomb(-d_left,golomb_m)

        if d_right >= 0:
            valRight = '0'+Golomb.to_golomb(d_right,golomb_m)
        else:
            valRight = '1'+Golomb.to_golomb(-d_right,golomb_m)

        if _ < 10:
            print(valLeft)
            print(valRight)
            print()

        buffer += BitArray(bin=valLeft) + BitArray(bin=valRight)

        if _ == 10:
            print(buffer.bin)

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

    outSamples = BitArray()

    print("Starting Golomb decoding.")
    # We don't know how many "soundframes" there are, so using sys.maxsize, it reads everything it can
    data = BitArray(sound.readframes(sys.maxsize)).bin
    sound.close()

    # TODO: Integrate Golomb decoding here. Create custom "for" to not loop every bit. ATM every bit is looped
    # twice. Plus comparing in "if"s, takes too long.
    c = int(math.ceil(math.log(golomb_m,2)))
    print("c: "+str(c))
    div = int(math.pow(2,c) - golomb_m)
    i = 0
    samplesDecoded = 0
    signValue = 1
    readingLeft = True
    valLeft = None
    valRight = None
    prevLeft = 0
    prevRight = 0
    pbar = tqdm(total=len(data))
    print(data[:512])
    while i < len(data):
        # Sign bit
        if data[i] == '1':
            signValue = -1
        else:
            signValue = 1
        if samplesDecoded <= 5:
            print("Sign: "+ str(data[i]))

        #Update index
        i += 1
        pbar.update(1)
        ### Start of Golomb coding
        # Unary coding
        pos = 0
        while data[i] == '1':
            i += 1
            pos += 1
            pbar.update(1)
        q = pos  #q -> '0'
        
        if samplesDecoded <= 5:
            print("Number of 1: "+ str(q))
        
        i +=1
        pbar.update(1)
        # Binary remainder
       
        #untill c-1 ((+1)because list slicing is right exclusive)

        if samplesDecoded <= 5:
            print("Bin: "+ str(data[i:i+c-1 +1]))

        if data[i:i+c-1 +1] == '1'*(c-1):
            #ler c, skip c-1
            if samplesDecoded <= 5:
                print(data[i:i+c+1])
            # If remainder is of length c, because of only c-1 '1' bits
            r = int(data[i:i+c +1],2)
            i += c
            pbar.update(c)
        else:
            #ler c-1, skip c-1-1
            if samplesDecoded <= 5:
                print(data[i:i+c-1 +1])
            r = int(data[i:i+c-1 +1],2)
            i += c-1
            pbar.update(c-1)

        if r > div:
            r = r - div
        val = (q*golomb_m + r)*signValue

        #first of next
        i += 1
        pbar.update(1)
        if samplesDecoded <= 5:
            print("first of next: "+data[i])


        if readingLeft:
            d_left = val
            valLeft = d_left+prevLeft
            prevLeft = valLeft
            readingLeft = False
        else:
            d_right = val
            valRight = d_right+prevRight
            prevRight = valRight
            readingLeft = True

        if valRight != None and valLeft != None:
            samplesDecoded += 1
            if samplesDecoded <= 5:
                print("left "+str(d_left))
                print("right "+str(d_right))
                print()
            outSamples += struct.pack('<hh', valLeft, valRight)
            valRight = None
            valLeft = None
    pbar.close()
        
    """
    for i,bit in data_iter:
        count += 1
        if(count%1000==0):
            print(str(int(count/1000))+" thousand bits")
        # Sign bit
        if signValue == None:
            if bit=='1':
                signValue = -1
            else:
                signValue = 1
        # Code bits
        else:
            if startGolomb < 0:
                startGolomb = i
            if bit=='0' or endBinary>-1:
                if endBinary > -1:
                    # End of unary
                    startBinary = i
                    endBinary = i+c
                elif(i < endBinary):
                    continue
                # End of Golomb word
                s = data[startGolomb:endBinary]
                #print(type(s))
                #print(s)

                val = Golomb.from_golomb(s,golomb_m) * signValue
                #print(val)
                if readingLeft:
                    valLeft = val
                else:
                    valRight = val
                # Value read. Reset control vars.
                startGolomb = -1
                startBinary = -1
                endBinary = -1
                signValue = None
                if valRight != None and valLeft != None:
                    outSamples += struct.pack('<hh', valLeft, valRight)
                    valRight = None
                    valLeft = None
    """

    # Write new file
    print("Done. Writing file...")
    out = wave.open(fname+"_decoded.wav", "w")
    out.setnchannels(n_channels)
    out.setsampwidth(samp_width)
    out.setframerate(framerate)
    out.writeframesraw(outSamples.bytes)
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

    #diff_encode(fname,512)
    #diff_decode(fname,512)

    #diff_encode(fname,1055)
    diff_decode(fname,1055)

if __name__ == "__main__":
    main()
