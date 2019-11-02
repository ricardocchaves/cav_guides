
import math 
from bitstring import BitArray

class Golomb():
  
    value = 0
    length = 0

    @classmethod
    def to_golomb(cls, val, m):
        
        q = val//m
        r = val%m

        if( (m & (m-1) == 0) and m != 0): 
            #power of 2
            r_binstr = "{0:b}".format(r)
        else:
            b = math.ceil(math.log2(m))
            x = r + 2**b - m
            
            r_binstr = "{0:b}".format(x)

        q_binstr = Golomb.to_unarystr(q)

        print(r_binstr)
        print(q_binstr)

        return q_binstr + r_binstr

    @classmethod 
    def to_unarystr(cls, val):
        return '1'*val + '0'
