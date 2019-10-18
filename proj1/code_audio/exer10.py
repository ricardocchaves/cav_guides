import wave, struct, sys, math

fileName = sys.argv[1]

sound = wave.open(fileName, "r")

fCMLeft = {}
fCMRight = {}
fCMMono = {}

print( "Number of channels",sound.getnchannels())
print ( "Sample width",sound.getsampwidth())
print ( "Frame rate.",sound.getframerate())
print ("Number of frames",sound.getnframes())
print ( "parameters:",sound.getparams())

prevLeft = None
prevRight = None
prevMono = None

for i in range(sound.getnframes()):
    sample = sound.readframes(1)
    valLeft, valRight = struct.unpack('<hh', sample )
    valMono = int( (valLeft + valRight) / 2 )

    #Histogram for Left
    if prevLeft:
        if prevLeft in fCMLeft.keys():
            if valLeft in fCMLeft[prevLeft].keys():
                fCMLeft[prevLeft][valLeft] += 1
            else:
                fCMLeft[prevLeft][valLeft] = 1
        else:
            fCMLeft[prevLeft] = {}
            fCMLeft[prevLeft][valLeft] = 1

    prevLeft = valLeft

    #Histogram for Right
    if prevRight:
        if prevRight in fCMRight.keys():
            if valRight in fCMRight[prevRight].keys():
                fCMRight[prevRight][valRight] += 1
            else:
                fCMRight[prevRight][valRight] = 1
        else:
            fCMRight[prevRight] = {}
            fCMRight[prevRight][valRight] = 1

    prevRight = valRight

    #Histogram for Mono
    if prevMono:
        if prevMono in fCMMono.keys():
            if valMono in fCMMono[prevMono].keys():
                fCMMono[prevMono][valMono] += 1
            else:
                fCMMono[prevMono][valMono] = 1
        else:
            fCMMono[prevMono] = {}
            fCMMono[prevMono][valMono] = 1

    prevMono = valMono

entLeft = 0
entRight = 0
entMono = 0

for prevValue in fCMLeft.keys():
    entropy = 0
    numberOfOccurrences = 0
    for val in fCMLeft[prevValue]:
        numberOfOccurrences += fCMLeft[prevValue][val]
        probVal = fCMLeft[prevValue][val] / len(fCMLeft[prevValue])
        entropy += probVal * math.log2(1/probVal)

    entLeft += ( numberOfOccurrences / sound.getnframes() ) * entropy

for prevValue in fCMRight.keys():
    entropy = 0
    numberOfOccurrences = 0
    for val in fCMRight[prevValue]:
        numberOfOccurrences += fCMRight[prevValue][val]
        probVal = fCMRight[prevValue][val] / len(fCMRight[prevValue])
        entropy += probVal * math.log2(1/probVal)

    entRight += ( numberOfOccurrences / sound.getnframes() ) * entropy

for prevValue in fCMMono.keys():
    entropy = 0
    numberOfOccurrences = 0
    for val in fCMMono[prevValue]:
        numberOfOccurrences += fCMMono[prevValue][val]
        probVal = fCMMono[prevValue][val] / len(fCMMono[prevValue])
        entropy += probVal * math.log2(1/probVal)

    entMono += ( numberOfOccurrences / sound.getnframes() ) * entropy

print(entLeft)
print(entRight)
print(entMono)

sound.close()
