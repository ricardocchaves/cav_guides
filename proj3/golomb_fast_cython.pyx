# compile this file: python3 setup_golomb.py build_ext --inplace
cdef class GolombFast_cython(object):
	cdef buffer
	cdef unsigned int byte, pointer, m, c, div
	def __init__(self,m,c,div):
		self.buffer = []

		self.byte = 0
		self.pointer = 7
		
		self.m = m
		self.c = c
		self.div = div

	cpdef toGolomb(self, unsigned int val):
		cdef int q,r,i
		q = val//self.m
		r = val % self.m
		
		for i in range(q):
			self._writeBit(1)
		
		self._writeBit(0)

		if (r<self.div):
			self._writeNBits(r,self.c-1)
		else:
			self._writeNBits(r,self.c)

	cpdef bytes getBytes(self):
		cdef bytes b
		b = bytes(self.buffer)
		self.buffer = []
		return b

	cdef _writeBit(self,unsigned char b):
		self.byte = self.byte | ((b & 0x01) << self.pointer)
		if (self.pointer > 0):
			self.pointer = self.pointer - 1
			return
		self.buffer.append(self.byte)
		self.pointer = 7
		self.byte = 0

	cdef _writeNBits(self,unsigned int b,unsigned int length):
		cdef int i
		cdef unsigned char bit
		i = length - 1
		while i>=0:
			bit = (b >> i%32) & 0x01
			self._writeBit(bit)
			i = i - 1