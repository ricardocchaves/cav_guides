import math

class Golomb():
    # Encode into Golomb
    # Val - int
    # m   - int
    @classmethod
    def to_golomb(cls, val, m):
        c = int(math.ceil(math.log(m,2)))
        r = val % m
        q =int(math.floor(val / m))
        div = int(math.pow(2,c) - m)
        unary = '1'*q

        if (r < div):
            b = c - 1
            binary = "{0:0" + str(b) + "b}"
            binary = binary.format(r)
        else:
            b = c
            binary = "{0:0" + str(b) + "b}"
            binary = binary.format(r+div)

        golomb = unary + "0" +str(binary)
        return golomb

    @classmethod 
    def _to_unarystr(cls, val):
        return '1'*val + '0'

    @classmethod
    def _from_unary(cls, val):
        return len(val[:-1])

    # Decode from Golomb
    # val - string
    @classmethod
    def from_golomb(cls, val, m):
        c = int(math.ceil(math.log(m,2)))
        div = int(math.pow(2,c) - m)
        pos = 0
        for bit in val:
            if bit == '0':
                break
            else:
                pos+=1
        q = pos
        r = int(val[pos+1:],2)
        if r > div:
            r = r - div
        return q*m + r

if __name__ == "__main__":
    for i in range(15+1):
        g = Golomb.to_golomb(i,5)
        print(g)
        print(Golomb.from_golomb(g,5))
        print("----")