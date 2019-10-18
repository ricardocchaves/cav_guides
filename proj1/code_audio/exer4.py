import wave, struct, sys, matplotlib.pyplot as plt

fileName = sys.argv[1]

sound = wave.open(fileName, "r")

histLeft = {}
histRight = {}
histMono = {}

print( "Number of channels",sound.getnchannels())
print ( "Sample width",sound.getsampwidth())
print ( "Frame rate.",sound.getframerate())
print ("Number of frames",sound.getnframes())
print ( "parameters:",sound.getparams())

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

plt.plot(list(histLeft.keys()), list(histLeft.values()), 'ro')
plt.show()

plt.plot(list(histRight.keys()), list(histRight.values()), 'ro')
plt.show()

plt.plot(list(histMono.keys()), list(histMono.values()), 'ro')
plt.show()

sound.close()
