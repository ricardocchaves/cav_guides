import wave
import struct

def printInfo(sound):
	print( "Number of channels",sound.getnchannels())
	print ( "Sample width",sound.getsampwidth())
	print ( "Frame rate.",sound.getframerate())
	print ("Number of frames",sound.getnframes())
	print ( "parameters:",sound.getparams())

# Copies an audio file
def copy(file_original,file_copy):
	original = wave.open(file_original, "r")
	copy = wave.open(file_copy, "w")

	printInfo(original)

	copy.setnchannels(original.getnchannels())
	copy.setsampwidth(original.getsampwidth())
	copy.setframerate(original.getframerate())

	for i in range(original.getnframes()):
	    sample = original.readframes(1)
	    copy.writeframesraw(sample)

	copy.close()
	original.close()

# returns left, right, mono
def getHistograms(fname,show=False):
	sound = wave.open(fname, "r")

	histLeft = {}
	histRight = {}
	histMono = {}

	printInfo(sound)

	for i in range(sound.getnframes()):
	    sample = sound.readframes(1)
	    valLeft, valRight = struct.unpack('<hh', sample )
	    valMono = int( (valLeft + valRight) / 2 )
	    if valLeft in histLeft.keys():
	        histLeft[valLeft] += 1
	    else:
	        histLeft[valLeft] = 0
	    if valRight in histRight.keys():
	        histRight[valRight] += 1
	    else:
	        histRight[valLeft] = 0
	    if valMono in histMono.keys():
	        histMono[valMono] += 1
	    else:
	        histMono[valMono] = 0

	if show:
		import matplotlib.pyplot as plt
		plt.figure("HistLeft")
		plt.plot(list(histLeft.keys()), list(histLeft.values()), 'ro')

		plt.figure("HistRight")
		plt.plot(list(histRight.keys()), list(histRight.values()), 'ro')

		plt.figure("HistMono")
		plt.plot(list(histMono.keys()), list(histMono.values()), 'ro')
		plt.show()

	sound.close()
	return histLeft,histRight,histMono

# Returns -1 if error
# TODO NOT WORKING
def quantize(fileNameOriginal,fileNameCopy,numberOfBitsNew):
	original = wave.open(fileNameOriginal, "r")
	copy = wave.open(fileNameCopy, "w")

	printInfo(original)
	copy.setnchannels(original.getnchannels())

	copy.setsampwidth(original.getsampwidth())
	numberOfBitsOriginal = original.getsampwidth() * 8
	if numberOfBitsNew >= numberOfBitsOriginal:
	    print("Number of bits should be less than the original file sample width")
	    return -1

	copy.setframerate(original.getframerate())

	for i in range(original.getnframes()):
	    sample = original.readframes(1)
	    valLeft, valRight = struct.unpack('<hh', sample )

	    newValLeft = int(valLeft / ((numberOfBitsOriginal - numberOfBitsNew)*2 ))
	    newValRight = int(valRight / ((numberOfBitsOriginal - numberOfBitsNew)*2 ))

	    newValLeft = newValLeft * 2 * (numberOfBitsOriginal - numberOfBitsNew)
	    newValRight = newValRight * 2 * (numberOfBitsOriginal - numberOfBitsNew)

	    sampleOut = struct.pack('<hh', newValLeft, newValRight)


	    copy.writeframesraw(sampleOut)
	    #value = struct.unpack('<i', sample )
	    #print(value)

	copy.close()
	original.close()

# Returns (SNR,mAE)
def SNR_MAE(fileNameOriginal, fileNameCopy):
	import math
	original = wave.open(fileNameOriginal, "r")
	copy = wave.open(fileNameCopy, "r")

	printInfo(original)

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

	if (sumOriginal - sumCopy) == 0:
		return 0,mAE
	else:
		snr = 10 * math.log10(sumOriginal / (sumOriginal - sumCopy))

	copy.close()
	original.close()

	return snr,mAE

# Returns (entLeft,entRight,entMono)
def entropy(fileName):
	import math

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

	#print(entLeft)
	#print(entRight)
	#print(entMono)

	sound.close()
	return entLeft,entRight,entMono

if __name__ == "__main__":
	entropy('../sounds/sample01.wav')