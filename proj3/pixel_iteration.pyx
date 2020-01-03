# compile this file: python3 setup.py build_ext --inplace
cpdef pixel_iteration_fast(int height,int width,unsigned char[:,:,:] frame):
	overalls = ([],[],[])
	for h in range(height):
		for w in range(width):
			y,u,v = frame[h,w]
			y_p, u_p, v_p = _predictor(frame,h,w)

			diff_y = y - y_p
			diff_u = u - u_p
			diff_v = v - v_p

			overalls[0].append(diff_y)
			overalls[1].append(diff_u)
			overalls[2].append(diff_v)
	return overalls

cdef _predictor(unsigned char[:,:,:]frame, int pos_h, int pos_w):
	cdef int y_p,u_p,v_p
	if pos_w-1>=0:
		a = frame[pos_h,pos_w-1]
	else:
		a = [0,0,0]

	if pos_h-1>=0:
		b = frame[pos_h-1,pos_w]
	else:
		b = [0,0,0]

	if pos_h-1>=0 and pos_w-1>=0:
		c = frame[pos_h-1,pos_w-1]
	else:
		c = [0,0,0]
	
	y_p = _nonLinearPredictor(a[0],b[0],c[0])
	u_p = _nonLinearPredictor(a[1],b[1],c[1])
	v_p = _nonLinearPredictor(a[2],b[2],c[2])
	return y_p,u_p,v_p

# c | b
# a | X
cdef int _nonLinearPredictor(int a, int b, int c):
	if c >= max(a,b):
		return min(a,b)
	elif c <= min(a,b):
		return max(a,b)
	else:
		return a+b-c