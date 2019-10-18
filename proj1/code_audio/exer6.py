import wave, struct, sys

fileNameOriginal = sys.argv[1]
fileNameCopy = sys.argv[2]
numberOfBitsNew = int(sys.argv[3])

original = wave.open(fileNameOriginal, "r")
copy = wave.open(fileNameCopy, "w")

print( "Number of channels",original.getnchannels())
copy.setnchannels(original.getnchannels())

print ( "Sample width",original.getsampwidth())
copy.setsampwidth(original.getsampwidth())
numberOfBitsOriginal = original.getsampwidth() * 8
if numberOfBitsNew >= numberOfBitsOriginal:
    print("Number of bits should be less than the original file sample width")
    exit()

print ( "Frame rate.",original.getframerate())
copy.setframerate(original.getframerate())

print ("Number of frames",original.getnframes())
print ( "parameters:",original.getparams())


for i in range(original.getnframes()):
    sample = original.readframes(1)
    valLeft, valRight = struct.unpack('<hh', sample )

    newValLeft = int(valLeft / ((numberOfBitsOriginal - numberOfBitsNew)*2 ))
    newValRight = int(valRight / ((numberOfBitsOriginal - numberOfBitsNew)*2 ))

    newValLeft = newValLeft * 2 * (numberOfBitsOriginal - numberOfBitsNew)
    newValRight = newValRight * 2 * (numberOfBitsOriginal - numberOfBitsNew)

    # if newValLeft > 127 or newValLeft < -128:
    #     print(newValLeft)
    # if newValRight > 127 or newValRight < -128:
    #     print(newValRight)

    sampleOut = struct.pack('<hh', newValLeft, newValRight)


    copy.writeframesraw(sampleOut)
    #value = struct.unpack('<i', sample )
    #print(value)

copy.close()
original.close()
