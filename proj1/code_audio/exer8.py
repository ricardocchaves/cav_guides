import wave, struct, sys, math

fileNameOriginal = sys.argv[1]
fileNameCopy = sys.argv[2]

original = wave.open(fileNameOriginal, "r")
copy = wave.open(fileNameCopy, "r")

print( "Number of channels",original.getnchannels())
print ( "Sample width",original.getsampwidth())
print ( "Frame rate.",original.getframerate())
print ("Number of frames",original.getnframes())
print ( "parameters:",original.getparams())

sumOriginal = 0
sumCopy = 0
mAE = 0

for i in range(original.getnframes()):
    sample = original.readframes(1)
    sampleCopy = copy.readframes(1)
    sampleVal = struct.unpack('<i', sample )[0]
    sampleCopyVal = struct.unpack('<i', sampleCopy)[0]

    if (sampleVal - sampleCopyVal) > mAE :          #Per sample or per channel?
        mAE = sampleVal - sampleCopyVal

    sumOriginal += sampleVal * sampleVal
    sumCopy += sampleCopyVal * sampleCopyVal

snr = 10 * math.log10(sumOriginal / (sumOriginal - sumCopy))

print(snr)
print(mAE)

copy.close()
original.close()
