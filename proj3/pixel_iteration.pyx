# compile this file: python3 setup.py build_ext --inplace
import numpy as np
cpdef decodeFrame(unsigned int height, unsigned int width, diffs):
	ch_pred = np.zeros((height,width))
	ch_frame = np.zeros((height,width))
	cdef int pred
	cdef int x,y

	for y in range(height):
		for x in range(width):
			pred = decode_predictor(x,y,ch_frame)
			ch_pred[y,x] = pred
			ch_frame[y,x] = pred + diffs[y,x]
	return ch_frame

cdef int decode_predictor(unsigned int x,unsigned int y, frame):
	cdef int a,b,c
	if x>0:
		a = int(frame[y,x-1])
	else:
		a = 0
	if y>0:
		b = int(frame[y-1,x])
	else:
		b = 0
	if y>0 and x>0:
		c = int(frame[y-1,x-1])
	else:
		c = 0
	
	return _nonLinearPredictor(a,b,c)

cdef int _nonLinearPredictor(int a,int b,int c):
	cdef int maxAB = max(a,b)
	cdef int minAB = min(a,b)
	if c >= maxAB:
		return minAB
	elif c <= minAB:
		return maxAB
	else:
		return a+b-c

####### Inter frame encoding.
from sys import maxsize
cpdef inter_frame_processing_fast(int height, int width, unsigned char[:,:,:] frame, unsigned char[:,:,:] ref_frame, int N, int offset):
	if not ((N != 0) and (N & (N-1) == 0)):
		print("N is not power of 2")
		return -1
	
	motion_vectors_to_encode = {}
	residual_diffs = {}

	for cblock_x in range(0,width//N):
		for cblock_y in range(0,height//N):
			#current block = (cblock_x, cblock_y)
			#print(cblock_x,cblock_y)
			for channel in range(0,3):
				#find best block for each channel
				diff = 0
				min_diff = maxsize
				best_block_x = -1
				best_block_y = -1
				best_diffs = -1

				for bx in range(cblock_x-offset, cblock_x+offset+1):
					for by in range(cblock_y-offset, cblock_y+offset+1):
						#Search for similar blocks in area of offset number of blocks around current block
						if bx >= 0 and bx < width//N and by>=0 and by<height//N:
							#This is a valid block to compare with
							#overall_diff, diffs = compare_blocks(frame, cblock_x,cblock_y, ref_frame, bx, by, N, channel)
							diffs = compare_blocks_fast(frame, cblock_x,cblock_y, ref_frame, bx, by, N, channel)
							overall_diff = sum(diffs)
							if abs(overall_diff) < abs(min_diff):
								min_diff = overall_diff
								best_diffs = diffs
								best_block_x = bx
								best_block_y = by

				#Compute motion vector for this channel
				motion_vector_x = best_block_x - cblock_x
				motion_vector_y = best_block_y - cblock_y

				motion_vectors_to_encode[cblock_x,cblock_y,channel] = (motion_vector_x,motion_vector_y)
				residual_diffs[cblock_x,cblock_y,channel] = best_diffs
					
	return motion_vectors_to_encode, residual_diffs

cpdef compare_blocks_fast(unsigned char[:,:,:] frame, int cbx, int cby, unsigned char[:,:,:]ref_frame, int bx, int by, int N, int ch):
	#convert block coordenates to pixel coordinates
	#return -1, [1]*16
	cdef int cbx_pixel_upper_left,cby_pixel_upper_left,bx_pixel_upper_left,by_pixel_upper_left
	cbx_pixel_upper_left = cbx * N
	cby_pixel_upper_left = cby * N
	
	bx_pixel_upper_left = bx * N
	by_pixel_upper_left = by * N

	cdef int pixel_value, reference_pixel_value, diff, overall_diff
	
	diffs = []
	for x in range(N):
		for y in range(N):

			pixel_value = <int>(frame[cby_pixel_upper_left + y, cbx_pixel_upper_left + x, ch])

			reference_pixel_value = <int>( ref_frame[by_pixel_upper_left + y, bx_pixel_upper_left + x, ch] ) 

			diff = pixel_value - reference_pixel_value 
			diffs += [diff]

	return diffs

import numpy as np
cpdef decode_inter_frame2_fast(int height,int width,unsigned char[:,:,:] ref_frame, int N, motion_vectors, residual_diffs):
	frame = np.array(ref_frame,copy=True)
	for cblock_x in range(0,width//N):
		for cblock_y in range(0,height//N):
			
			#get upper left pixel of this block
			block_pixel_upper_left_x = cblock_x * N
			block_pixel_upper_left_y = cblock_y * N
			
			for channel in range(0,3):
				#go get reference block
				ref_block_x = cblock_x + motion_vectors[cblock_x,cblock_y,channel][0] #mv.x
				ref_block_y = cblock_y + motion_vectors[cblock_x,cblock_y,channel][1] #mv.y
				#convert block coordinated to pixel coordinates
				#get upper left pixel of this reference block
				ref_block_pixel_upper_left_x = ref_block_x * N
				ref_block_pixel_upper_left_y = ref_block_y * N

				#get residuals for this block (list with residuals for each coordinate)
				block_residuals = residual_diffs[cblock_x,cblock_y,channel]
				
				#get pixel values from reference block and add residuals
				#and write to frame (that is being contructed)
				#iterating inside the block
				for x in range(0,N):
					for y in range(0,N):
						frame[block_pixel_upper_left_y + y, block_pixel_upper_left_x + x, channel] = \
							ref_frame[ref_block_pixel_upper_left_y + y, ref_block_pixel_upper_left_x + x, channel] \
								+ block_residuals[x*N + y]

	return frame                 