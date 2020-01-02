import numpy as np
# compile this file: python3 setup.py build_ext --inplace
cpdef unsigned char[:,:,:] pixel_iteration_fast(int height, int width, unsigned char [:,:,:] frame):
	cdef unsigned char[:,:,:] outFrame
	cdef int h, w
	cdef unsigned char y,u,v,y_p,u_p,v_p

	outFrame = np.copy(frame)
	for h in range(height):
		for w in range(width):
			y,u,v = frame[h,w]
			y_p = y
			u_p = u
			v_p = v
			outFrame[h,w,0] = y_p
			outFrame[h,w,1] = u_p
			outFrame[h,w,2] = v_p
	return outFrame