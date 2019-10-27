import sys
sys.path.append('..') # Making the other packages in the repository visible."

import proj1.imgpy as img
import proj1.audiopy as audio

def main():
    print("Running")
    #img.show('../fotos/kodim03.png')

def testingBitstring():
    from bitstring import BitArray
    """
    ## Creating BitArray
    # from a binary string
    a = BitArray('0b001')
    # from a hexadecimal string
    b = BitArray('0xff470001')
    # straight from a file
    c = BitArray(filename='somefile.ext')
    # from an integer
    d = BitArray(int=540, length=11)
    # using a format string
    d = BitArray('int:11=540')
    """
    a = BitArray('0xff01')
    b = BitArray('0b11001000')
    print(b.bin)
    print(b.hex)
    print(b.bytes)
    print(b.len)
    print(b.length)
    print(b+b)
    print(a+(b+b))

if __name__ == "__main__":
    testingBitstring()