import math
from numpy import binary_repr

class GolombFast():
	def __init__(self,m,c,div):
		self.buffer = []

		self.byte = 0
		self.pointer = 7
		
		self.m = m
		self.c = c
		self.div = div

	def toGolomb(self,val):
		q = val//self.m
		#r = val-q*self.m
		r = val % self.m

		for _ in range(q):
			self._writeBit(1)
		
		self._writeBit(0)

		if (r<self.div):
			self._writeNBits(r,self.c-1)
		else:
			self._writeNBits(r,self.c)

	def getBytes(self):
		b = bytes(self.buffer)
		self.buffer = []
		return b

	def _writeBit(self,b):
		self.byte |= (b & 0x01) << self.pointer
		if (self.pointer > 0):
			self.pointer-=1
			return
		self.buffer.append(self.byte)
		self.pointer = 7
		self.byte = 0

	def _writeNBits(self,b,length):
		i = length-1
		while i>=0:
			bit = (b >> i%32) & 0x01
			self._writeBit(bit)
			i-=1

if __name__ == "__main__":
	import time
	def to_golomb_fast(val,m,c,div):
		r = val % m
		q =int(math.floor(val / m))
		unary = "1"*q

		if (r < div):
			b = c - 1
			binary = binary_repr(r,width=b)
		else:
			b = c
			binary = binary_repr(r,width=b)

		return int(unary+"0"+binary,2)

	g = GolombFast(2,2,2)
	N = 1000000

	print("## Using LIST OF BYTES")
	print("Starting {} golomb...".format(N))
	s = time.time()
	for i in range(N):
		g.toGolomb(9)
	t = time.time()-s
	print("Finished: {:.2f}s. {}/s".format(t,N//t))

	print("Starting conversion to bytes...")
	s = time.time()
	print(g.buffer[:10])
	b = g.getBytes()
	print(len(b))
	print("{}...".format(b[:15]))
	t = time.time()-s
	print("Finished: {:.2f}s.".format(t))

	
	print("\n## Using STRING")
	print("Starting {} golomb...".format(N))
	s = time.time()
	buf = []
	for i in range(N):
		buf.append(to_golomb_fast(9,2,2,2))
	t = time.time()-s
	print("Finished: {:.2f}s. {}/s".format(t,N//t))
	
	print("Starting conversion to bytes...")
	s = time.time()
	#b = ''.join(buf).encode('utf-8')
	b = bytes(buf)
	print(buf[:10])
	print(len(b))
	print("{}...".format(b[:15]))
	t = time.time()-s
	print("Finished: {:.2f}s.".format(t))