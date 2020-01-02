
import cv2 as cv
import sys
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Value
import time
from pixel_iteration import pixel_iteration_fast

from VideoCaptureYUV import VideoCaptureYUV

import sys
sys.path.append('..') # Making the other packages in the repository visible.


def main():
    inputFile = sys.argv[1]
    cap = VideoCaptureYUV(inputFile)

    ret, ref_frame = cap.read_raw()
    if not ret:
        input("NOT RET")

    ret, frame = cap.read_raw()
    if not ret:
        input("NOT RET 2")

    N = 8
    offset = 1
    print(cap.height)
    print(cap.width)
    #print(type(frame))
    motion_vectors_to_encode, residual_diffs = inter_frame_processing(cap.height,cap.width,frame, ref_frame, N, offset)
    #written is this order:
    #                       left to right, top to bottom, blocks first
    #                       left to right, top to bottom, pixel diffs first
    #                       Y first then U, then V 

    #example: motion vectors = MV bloco 0,0 para Y
    #                           MV bloco 0,0 para U
    #                           MV bloco 0,0 para V
    #                           MV bloco 0,1 para Y
    #                           ...
    
    #example para blocos 2x2 (N=2)
    #
    #                      (0,0) = valor da diferença para o bloco estimado, do pixel 0,0 
    #                        ^
    #     residual_diffs = (0,0) - (0,1) - (1,0) - (1,1) | 0,0 - 0,1 - 1,0 - 1,1 | 0,0 - 0,1 - 1,0 - 1,1 | 0,0 - 0,1 - 1,0 - 1,1 |
    #                           bloco 0,0 Y                    bloco = 0,0 U           bloco 0,0 V            bloco 0,1 Y

    #print(motion_vectors_to_encode)
    decode_inter_frame(cap.height,cap.width,ref_frame,N, motion_vectors_to_encode, residual_diffs)

def decode_inter_frame(height,width,ref_frame, N, motion_vectors, residual_diffs):

    frame = ref_frame #TODO: change to create new np array
    for cblock_x in tqdm(range(0,width//N)):
        for cblock_y in range(0,height//N):
            
            block_index = cblock_y + N*cblock_x

            mv_index = block_index*3
            block_mv_y = motion_vectors[mv_index]
            block_mv_u = motion_vectors[mv_index+1]
            block_mv_v = motion_vectors[mv_index+2]
            block_mv_array = [block_mv_y, block_mv_u, block_mv_v ]

            residual_index = block_index*N*N*3
            block_residuals = residual_diffs[residual_index:residual_index+N*N*3 ] #residuais para os 3 vetores y,u,v deste bloco
          
            #Fill in this block into the frame
            block_pixel_upper_left_x = cblock_x * N
            block_pixel_upper_left_y = cblock_y * N
            
            for x in range(0,N):
                for y in range(0,N):
                    
                    for channel in range(0,3):
                        #go get reference block
                        ref_block_x = cblock_x + block_mv_array[channel][0] #mv.x
                        ref_block_y = cblock_y + block_mv_array[channel][1] #mv.y
                        #convert block coordinated to pixel coordinates
                        ref_block_pixel_upper_left_x = ref_block_x * N
                        ref_block_pixel_upper_left_y = ref_block_y * N

                        #get pixel values from reference bloc and add residuals
                        #and write to frame (that is being contructed)
                        for x in range(0,N):
                            for y in range(0,N):
                                frame[block_pixel_upper_left_y + y, block_pixel_upper_left_x + x, channel] = \
                                    ref_frame[ref_block_pixel_upper_left_y + y, ref_block_pixel_upper_left_x + x, channel] \
                                        + block_residuals[channel*N*N + x*N + y]

    return frame
                                

def inter_frame_processing(height, width, frame, ref_frame, N, offset):
    if not ((N != 0) and (N & (N-1) == 0)):
        print("N is not power of 2")
        return -1
    
    motion_vectors_to_encode = []
    residual_diffs = []

    for cblock_x in tqdm(range(0,width//N)):
        for cblock_y in range(0,height//N):
            #current block = (cblock_x, cblock_y)
            #print(cblock_x,cblock_y)
            for channel in range(0,3):
                #find best block for each channel
                diff = 0
                min_diff = sys.maxsize
                best_block_x = -1
                best_block_y = -1
                best_diffs = -1

                for bx in range(cblock_x-offset, cblock_x+offset+1):
                    for by in range(cblock_y-offset, cblock_y+offset+1):
                        #Search for similar blocks in area of offset number of blocks around current block
                        if bx >= 0 and bx < width//N and by>=0 and by<height//N:
                            #This is a valid block to compare with
                            overall_diff, diffs = compare_blocks(frame, cblock_x,cblock_y, ref_frame, bx, by, N, channel)
                            #print(abs(overall_diff))
                            #print(bx,by)
                            if abs(overall_diff) < abs(min_diff):
                                min_diff = overall_diff
                                best_diffs = diffs
                                best_block_x = bx
                                best_block_y = by

                #Compute motion vector for this channel
                motion_vector_x = best_block_x - cblock_x
                motion_vector_y = best_block_y - cblock_y
                #print(cblock_x,cblock_y)
                #print(best_block_x,best_block_y)
                #print(min_diff)
                #print("MV: "+str(motion_vector_x)+" , "+str(motion_vector_y))
                #print("best diffs = "+ str(best_diffs))
                #input()

                motion_vectors_to_encode += [(motion_vector_x,motion_vector_y)]
                residual_diffs += best_diffs
                

    return motion_vectors_to_encode, residual_diffs


def compare_blocks(frame, cbx, cby, ref_frame, bx, by, N, ch):
    #convert block coordenates to pixel coordinates
    cbx_pixel_upper_left = cbx * N
    cby_pixel_upper_left = cby * N
    
    bx_pixel_upper_left = bx * N
    by_pixel_upper_left = by * N
    
    overall_diff = 0
    diffs = []
    for x in range(0,N):
        for y in range(0,N):

            pixel_value = int( frame[cby_pixel_upper_left + y, cbx_pixel_upper_left + x, ch] )

            reference_pixel_value = int( ref_frame[by_pixel_upper_left + y, bx_pixel_upper_left + x, ch] ) 

            diff = pixel_value - reference_pixel_value
            overall_diff += diff
            diffs += [diff]

    return overall_diff, diffs


if __name__ == "__main__":
	main()