import wave, struct, sys

fileNameOriginal = sys.argv[1]
fileNameCopy = sys.argv[2]

original = wave.open(fileNameOriginal, "r")
copy = wave.open(fileNameCopy, "w")

print( "Number of channels",original.getnchannels())
copy.setnchannels(original.getnchannels())
print ( "Sample width",original.getsampwidth())
copy.setsampwidth(original.getsampwidth())
print ( "Frame rate.",original.getframerate())
copy.setframerate(original.getframerate())
print ("Number of frames",original.getnframes())
print ( "parameters:",original.getparams())

for i in range(original.getnframes()):
    sample = original.readframes(1)
    copy.writeframesraw(sample)
    #value = struct.unpack('<i', sample )
    #print(value)

copy.close()
original.close()
