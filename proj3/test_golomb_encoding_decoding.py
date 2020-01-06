from golomb_cython import GolombCython
from golomb_cython import ReadGolomb
import time
import math

#N = 1*1000*1000
N = 10
ms = [4,5,13]
#ms = [5]

for m in ms:
    c = int(math.ceil(math.log(m,2)))
    div = int(math.pow(2,c) - m)
    enc = GolombCython(m,c,div)
    print("\nStarting {} golomb, m={}...".format(N,m))
    s = time.time()
    for i in range(N):
        enc.toGolomb(i)
    t = time.time()-s
    print("Finished: {:.2f}s. {}/s".format(t,N//t))
    encoded = enc.getBytes()
    print(list(encoded))

    print("Decoding...")
    s = time.time()
    dec = ReadGolomb(m,encoded)
    vals = dec.getValues()
    #print(len(vals))
    print(vals)
    t = time.time()-s
    print("Finished: {:.2f}s. {}/s".format(t,N//t))

"""
g = GolombFast(2,2,2)
#N = 47923200

print("## Using LIST OF BYTES")
print("Starting {} golomb...".format(N))
s = time.time()
for i in range(N):
    g.toGolomb(9)
t = time.time()-s
print("Finished: {:.2f}s. {}/s".format(t,N//t))

print("Starting conversion to bytes...")
s = time.time()
b = g.getBytes()
#print(len(b))
#print("{}...".format(bin(int.from_bytes(b[:4],'big'))))
t = time.time()-s
print("Finished: {:.2f}s.".format(t))

print("## Using CYTHON")
from golomb_cython import GolombFast_cython
g = GolombFast_cython(2,2,2)
print("Starting {} golomb...".format(N))
s = time.time()
for i in range(N):
    g.toGolomb(9)
t = time.time()-s
print("Finished: {:.2f}s. {}/s".format(t,N//t))

print("Starting conversion to bytes...")
s = time.time()
b = g.getBytes()
#print(len(b))
#print("{}...".format(bin(int.from_bytes(b[:4],'big'))))
t = time.time()-s
print("Finished: {:.2f}s.".format(t))
"""