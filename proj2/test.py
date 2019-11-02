from BitSet import Bitset
from Golomb import Golomb


val = 13
m = 5
print( "Value: " + str(val)+" - M = "+str(m))
x = Golomb.to_golomb(val,m)
print(x)
