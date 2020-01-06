# compile this file: python3 setup_golomb.py build_ext --inplace
from libc.math cimport ceil,log2,pow
cdef class GolombCython(object):
	cdef buffer
	cdef unsigned int pointer, m, c, div
	cdef unsigned char byte
	def __init__(self,unsigned int m,unsigned int c,unsigned int div):
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
			#self._writeNBits(r,self.c)
			self._writeNBits(r+self.div,self.c)

	cpdef bytes getBytes(self):
		cdef bytes b
		# Padding
		if self.pointer != 7:
			#self._writeNBits(0,self.pointer+1)
			#self._writeNBits(0xff,7-self.pointer)
			self._writeNBits(0xff,self.pointer+1)

		b = bytes(self.buffer)
		self.buffer = []
		return b

	cdef _writeBit(self, unsigned char b):
		self.byte |= ((b & 0x01) << self.pointer)
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




cdef class ReadGolomb(object):
	cdef list values, buffer
	cdef unsigned int m, c, div, next_read, end
	cdef unsigned char byte
	cdef int pointer
	def __init__(self,unsigned int m,raw_data):
		self.buffer = list(raw_data)

		self.byte = 0
		self.pointer = -1
		self.next_read = 0
		self.end = 0
		
		self.m = m
		self.c = int(ceil(log2(self.m)))
		self.div = int(pow(2,self.c) - self.m)

		self.values = self.processBuffer()

	cdef processBuffer(self):
		cdef values = []
		cdef unsigned int val
		#while self.next_read < len(self.buffer):
		while self.end == 0 or self.pointer>0:
			val = self.readVal()
			#if self.end == 1:
				#break
			if self.pointer >= -1:
				values += [val]
		return values

	cdef unsigned int readVal(self):
		cdef unsigned int q,r,tmp_r
		cdef unsigned char bit
		q = 0
		r = 0

		while(self.readBit()==1):
			q += 1

		tmp_r = self.readNBits(self.c-1)
		if tmp_r >= self.div:
			bit = self.readBit()
			r = tmp_r << 1
			r = r | bit
		else:
			r = tmp_r
		
		if r >= self.div:
			r = r - self.div
		return self.decode(q,r)

	cdef unsigned char readBit(self):
		cdef unsigned char bit
		if (self.pointer < 0) and (self.end == 0):
			self.byte = self.buffer[self.next_read]
			self.next_read = self.next_read + 1
			if(self.next_read >= len(self.buffer)):
				self.end = 1
			self.pointer = 7
		bit = (self.byte >> self.pointer) & 0x01
		self.pointer = self.pointer - 1
		return bit

	cdef unsigned int readNBits(self,unsigned int n):
		cdef unsigned int value = 0
		for i in range(n):
			value = value << 1 | self.readBit()
		return value

	cdef unsigned int decode(self, unsigned int q, unsigned int r):
		return q*self.m+r

	cpdef getValues(self):
		return self.values
