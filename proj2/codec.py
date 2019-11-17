import wave
import struct
import sys
from Golomb import Golomb
from tqdm import tqdm
from bitstring import BitArray
import math

### Golomb encoding function
# outFname - name of the output encoded fiel
# [left|right]Samples - lists of samples, eg: [3342,-2213,98,1233,...]
# [left|right]_symbolToValue - dictionary mapping symbols to values, eg: {0:-223,1:930,2:9352,...}
# [left|right]_valueToSymbol - dictionary mapping values to symbols, eg: {-223:0,930:1,9352:2,...}
# m_[left|right] - m value of each channel. To be used in the Golomb encoding
# n_channels, samp_width, framerate - attributes of the WAVE file
def encode(outFname,leftSamples,rightSamples,left_symbolToValue,
    left_valueToSymbol,right_symbolToValue, right_valueToSymbol, m_left, m_right,
    n_channels, samp_width, framerate):

    buffer = BitArray()

    c_left = int(math.ceil(math.log(m_left,2)))
    div_left = int(math.pow(2,c_left) - m_left)
    c_right = int(math.ceil(math.log(m_right,2)))
    div_right = int(math.pow(2,c_right) - m_right)
    
    print("Golomb encoding {} left samples, {} right samples.".format(len(leftSamples),len(rightSamples)))
    # assuming the number of samples on both channels is equal
    for i in tqdm(range(len(leftSamples))):
        leftSample = leftSamples[i]
        rightSample = rightSamples[i]

        leftSample = left_valueToSymbol[leftSample]
        rightSample = right_valueToSymbol[rightSample]

        valLeft = Golomb.to_golomb(leftSample,m_left,c_left,div_left)
        valRight = Golomb.to_golomb(rightSample,m_right,c_right,div_right)

        buffer += BitArray(bin=valLeft) + BitArray(bin=valRight)

    print("Done. Writing file...")
    # Write new file
    out = wave.open(outFname, "w")
    out.setnchannels(n_channels)
    out.setsampwidth(samp_width)
    out.setframerate(framerate)

    # Adding headers
    header = {"left_symbolToValue":left_symbolToValue,"right_symbolToValue":right_symbolToValue,
    "m_left":m_left,"m_right":m_right}

    from pickle import dumps
    header = dumps(header) # bytes
    header_size = len(header)

    out.writeframesraw(struct.pack('<i',header_size))
    out.writeframesraw(header)

    print("Wrote header with size {} bytes.".format(header_size))

    # Writing data
    out.writeframesraw(buffer.tobytes())
    out.close()
    print("File written.")

### Golomb decoding function
### Opens the file fname, gets the headers required for the decoding process and writes the
### decoded WAVE file to outFname
def decode(fname, outFname):
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
    header_serialized = sound.readframes(header_size)
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
            d_left = left_symbolToValue[val]
            valLeft = d_left+prevLeft
            prevLeft = valLeft
            readingLeft = False
        else:
            d_right = right_symbolToValue[val]
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
    out.writeframesraw(outSamples.tobytes())
    out.close()
    print("File written: {}".format(outFname))

### Golomb encoding function calculating differences between samples.
# fname - file to be read
# golomb_m - m value to be used in Golomb
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
    c = int(math.ceil(math.log(golomb_m,2)))
    div = int(math.pow(2,c) - golomb_m)
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
            valLeft = '0'+Golomb.to_golomb(d_left,golomb_m,c,div)
        else:
            valLeft = '1'+Golomb.to_golomb(-d_left,golomb_m,c,div)

        if d_right >= 0:
            valRight = '0'+Golomb.to_golomb(d_right,golomb_m,c,div)
        else:
            valRight = '1'+Golomb.to_golomb(-d_right,golomb_m,c,div)

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
    out = wave.open(fname+"_encoded.wav".format(golomb_m), "w")
    out.setnchannels(n_channels)
    out.setsampwidth(samp_width)
    out.setframerate(framerate)
    out.writeframesraw(buffer.tobytes())
    out.close()
    print("File written.")

### Golomb decoding function. Complement of diff_encode
# fname - file to be read
# golomb_m - m value to be used in Golomb
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

    log_m = math.log(golomb_m,2)
    c = int(math.ceil(log_m))
    powerOfTwo = log_m == c
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
        if samplesDecoded <= 10:
            print("Sign: "+ str(data[i]))

        #Update index
        i += 1
        pbar.update(1)
        ### Start of Golomb coding
        # Unary coding
        pos = 0
        #print("data[i] before: "+data[i])
        if i>=len(data):
            break
        while data[i] == '1':
            i += 1
            pos += 1
            #print("data[i] while: "+data[i])
            pbar.update(1)
        q = pos  #q -> '0'
        #print("data[i] after: "+data[i])
        
        if samplesDecoded <= 10:
            print("Number of 1: "+ str(q))
        
        i +=1
        pbar.update(1)
        # Binary remainder
       
        #untill c-1 ((+1)because list slicing is right exclusive)

        if i+c-1>=len(data):
            break
        if samplesDecoded <= 10:
            print("Bin: "+ str(data[i:i+c-1]))

        if int(data[i:i+c-1],2)>=div:
            #ler c, skip c-1
            if i+c>=len(data):
                break
            if samplesDecoded <= 10:
                print(data[i:i+c])
            # If remainder is of length c, because of only c-1 '1' bits
            r = int(data[i:i+c],2)
            i += c
            pbar.update(c)
        else:
            #ler c-1, skip c-1-1
            if samplesDecoded <= 10:
                print(data[i:i+c-1])
            r = int(data[i:i+c-1],2)
            i += c-1
            pbar.update(c-1)

        if r >= div:
            r = r - div
        val = (q*golomb_m + r)*signValue

        if samplesDecoded <= 10:
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
            if samplesDecoded <= 10:
                print("left "+str(d_left))
                print("right "+str(d_right))
                print()
            samplesDecoded += 1
            outSamples += struct.pack('<hh', valLeft, valRight)
            valRight = None
            valLeft = None
    pbar.close()

    # Write new file
    print("Done. Writing file...")
    out = wave.open(fname+"_decoded.wav", "w")
    out.setnchannels(n_channels)
    out.setsampwidth(samp_width)
    out.setframerate(framerate)
    out.writeframesraw(outSamples.tobytes())
    out.close()
    print("File written.")

def main():
    if len(sys.argv) < 2:
        print("USAGE: python3 {} sound_file.wav".format(sys.argv[0]))
        return
    fname = sys.argv[1]

    diff_encode(fname,512)
    #diff_decode(fname,512)

    #diff_encode(fname,1055)
    #diff_decode(fname,1055)

if __name__ == "__main__":
    main()
