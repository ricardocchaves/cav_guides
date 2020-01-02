
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

    N = 4
    offset = 8
    print(cap.height)
    print(cap.width)
    
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

def inter_frame_processing(height, width, frame, ref_frame, N, offset):
    if not ((N != 0) and (N & (N-1) == 0)):
        print("N is not power of 2")
        return -1
    
    motion_vectors_to_encode = []
    residual_diffs = []

    for cblock_x in range(0,width//N):
        for cblock_y in range(0,height//N):
            #current block = (cblock_x, cblock_y)

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
                        if bx >= 0 and bx < width and by>=0 and by<height:
                            #This is a valid block to compare with
                            overall_diff, diffs = compare_blocks(frame, cblock_x,cblock_y, ref_frame, bx,by, N, channel)
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

                motion_vectors_to_encode +=(motion_vector_x,motion_vector_y)
                residual_diffs += [best_diffs]
                

    return motion_vectors_to_encode, residual_diffs


def compare_blocks(frame, cbx, cby, ref_frame, bx,by, N, ch):
    #convert block coordenates to pixel coordinates
    cbx_pixel_upper_left = cbx * N
    cby_pixel_upper_left = cby * N
    
    bx_pixel_upper_left = bx * N
    by_pixel_upper_left = by * N
    
    overall_diff = 0
    diffs = []
    for x in range(0,N):
        for y in range(0,N):

            pixel_value = int( frame[cbx_pixel_upper_left + x, cby_pixel_upper_left + y, ch] )

            reference_pixel_value = int( ref_frame[bx_pixel_upper_left + x,by_pixel_upper_left + y, ch] ) 

            overall_diff += pixel_value - reference_pixel_value
            diffs += [pixel_value - reference_pixel_value]

    return overall_diff, diffs


if __name__ == "__main__":
	main()