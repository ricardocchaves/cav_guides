
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

    print(cap.height)
    print(cap.width)
    print(type(frame))
    print(cap.height_chroma)
    print(cap.width_chroma)
    
    #############################################################################################
    #ENCODE

    #########1##########
    #get frame to encode and corresponant reference frame
    #frame e ref_frame
    
    #########2##########
    #choose encoding parameters and encode
    N = 8 #size of block (N*N) on luma
    offset = 2 #radius of the neighbourhood for reference block search
    
    motion_vectors_to_encode, residual_diffs = inter_frame_processing(cap.height,cap.width,cap.height_chroma,cap.width_chroma,frame, ref_frame, N, offset)

    #print(motion_vectors_to_encode)
    #motion_vectors_to_encode[x,y] = (vector_x,vector_y) -> tuplo representa MV = (vec_x,vec_y) para um bloco x,y
    #dicionario -> { (block_x,block_y) : (vec_x,vec_y), (block_x,block_y) : (vec_x,vec_y),...}

    #print(residual_diffs[0,0,0])
    #residual_diffs[cblock_x,cblock_y,channel] = matriz das diferenças de um bloco
    
    #print(motion_vectors_to_encode[2,3])
    #print(residual_diffs[2,3,1])

    #########3##########
    #tranform motion vectors and residuals to lists of numbers to encode with golomb
    values_to_encode = dict()
    frame_num = 0

        #transform MV
    mv_aslist = mv_tolist(motion_vectors_to_encode)

        #transform residuals
    residuals_aslist = residual_diffs_tolist(residual_diffs,cap.height,cap.width,N)

        #save both on values to encode data structure
    values_to_encode[frame_num] = (mv_aslist,residuals_aslist)
    
    #apply golomb and end frame encoding
    #######################################################################################
    
    #DECODE

    #########1##########
    #read values from golomb
    mv_aslist,residuals_aslist = values_to_encode[frame_num]

    #########2##########
    #convert MV list back to the usual dictionary of motion_vectors[block_x,block_y]
    mv_to_decode = mv_fromlist(mv_aslist,cap.height,cap.width,N)

    #convert resuduals list back to the usual dictionary of residual_diff[block_x,block_y,channel]
    residuals_to_decode = residuals_fromlist(residuals_aslist,cap.height,cap.width,cap.height_chroma, cap.width_chroma,N)

    #print(mv_to_decode[2,3])
    #print(residuals_to_decode[2,3,1])
    
    #########3##########
    #retrieve frame using the converted dictionaries
    decoded_frame = decode_inter_frame2(cap.height,cap.width,cap.height_chroma,cap.width_chroma,ref_frame,N, mv_to_decode, residuals_to_decode)
    
    #END of decoding
    #################################################################################################################################################

    #Just random testing
    print("----")
    x= 556
    y= 23
    
    frame_ch = frame.getMatrix(1)
    print(decoded_frame[y,x,1])
    print(frame_ch[y,x])
    print("----")
    print("----")
    x= 234
    y= 49
    print(decoded_frame[y,x,1])
    print(frame_ch[y,x])
    print("----")
    print("----")
    x= 624
    y= 309
    print(decoded_frame[y,x,1])
    print(frame_ch[y,x])
    print("----")
    print("----")
    x= 10
    y= 10
    print(decoded_frame[y,x,1])
    print(frame_ch[y,x])
    print("----")
    
def mv_tolist(motion_vectors):
    mv_aslist = list(sum(motion_vectors.values(), ()))
    #print(mv_aslist)
    return mv_aslist

def mv_fromlist(motion_vectors, height,width,N):
    x = True
    x_value = 0
    y_value = 0
    mv_result = {}
    i = 0
    for cblock_x in tqdm(range(0,width//N)):
        for cblock_y in range(0,height//N):
            x_value = motion_vectors[i]
            y_value = motion_vectors[i+1]
            mv_result[cblock_x,cblock_y] = (x_value,y_value)
            i += 2
 
    return mv_result


def residual_diffs_tolist(residual_diffs,height,width,N):
    residuals_aslist = []
    for cblock_x in tqdm(range(0,width//N)):
        for cblock_y in range(0,height//N):
            for channel in range(0,3):

                diff_block = residual_diffs[cblock_x,cblock_y,channel]
                residuals_aslist += list(diff_block.flatten())
        
    #print(residuals_aslist)
    return residuals_aslist

def residuals_fromlist(residuals,height,width,height_chroma,width_chroma,N):
    i = 0
    residuals_result = {}
    for cblock_x in tqdm(range(0,width//N)):
        for cblock_y in range(0,height//N):
            for channel in range(0,3):

                if channel == 0:
                    block_size_x = N
                    block_size_y = N
                else:
                    chroma_height_ratio = height/height_chroma
                    chroma_width_ratio = width/width_chroma
                    block_size_x = int(N//chroma_height_ratio)
                    block_size_y = int(N//chroma_width_ratio)

                block_residuals = np.asarray(residuals[i:i+block_size_x*block_size_y])
                residuals_result[cblock_x,cblock_y,channel] =  block_residuals.reshape((block_size_x,block_size_y))
                i = i + block_size_x*block_size_y
    
    return residuals_result

        

def decode_inter_frame(height,width,ref_frame, N, motion_vectors, residual_diffs):

    frame = np.ndarray(shape=(height,width)) #TODO: change to create new np array
    
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
            
            #iterate inside the block
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


def decode_inter_frame2(height,width,height_chroma,width_chroma,ref_frame, N, motion_vectors, residual_diffs):

    #frame = ref_frame #TODO: change to create new np array
    frame = np.ndarray(shape=(height,width,3),dtype=np.uint8)

    for cblock_x in tqdm(range(0,width//N)):
        for cblock_y in range(0,height//N):
            
            #get luma to chroma block size ratio
            chroma_height_ratio = height/height_chroma
            chroma_width_ratio = width/width_chroma
            chroma_block_height = int(N//chroma_height_ratio)
            chroma_block_width = int(N//chroma_width_ratio)
            
            #get upper left pixel of this block for luma 
            block_pixel_upper_left_x = cblock_x * N
            block_pixel_upper_left_y = cblock_y * N
            #get upper left pixel of this block for chroma 
            chroma_block_pixel_upper_left_x = cblock_x * chroma_block_width
            chroma_block_pixel_upper_left_y = cblock_y * chroma_block_height
        
            #get reference block using MV
            ref_block_x = cblock_x + motion_vectors[cblock_x,cblock_y][0] #mv.x
            ref_block_y = cblock_y + motion_vectors[cblock_x,cblock_y][1] #mv.y

            #get upper left pixel of this reference block for luma
            ref_block_pixel_upper_left_x = ref_block_x * N
            ref_block_pixel_upper_left_y = ref_block_y * N
            #get upper left pixel of this reference block for chroma
            ref_chroma_block_pixel_upper_left_x = ref_block_x * chroma_block_width
            ref_chroma_block_pixel_upper_left_y = ref_block_y * chroma_block_height
            
            for channel in range(0,3):
                if channel == 0:
                    #Luma blocks
                    #get block from ref frame
                    ref_block = ref_frame.getMatrix(channel)[ref_block_pixel_upper_left_y:ref_block_pixel_upper_left_y + N, ref_block_pixel_upper_left_x:ref_block_pixel_upper_left_x + N]
            
                    #get residuals for this block (matrix N*N with residuals)
                    block_residuals = residual_diffs[cblock_x,cblock_y,channel]
                
                    #Compute frame block
                    frame[block_pixel_upper_left_y:block_pixel_upper_left_y+N, block_pixel_upper_left_x:block_pixel_upper_left_x + N, channel] = \
                        ref_block + block_residuals
                else:
                    #chroma blocks
                    #get block from ref frame
                    ref_block = ref_frame.getMatrix(channel)[ref_chroma_block_pixel_upper_left_y:ref_chroma_block_pixel_upper_left_y + chroma_block_height, ref_chroma_block_pixel_upper_left_x:ref_chroma_block_pixel_upper_left_x + chroma_block_width]
            
                    #get residuals for this block (matrix N*N with residuals)
                    block_residuals = residual_diffs[cblock_x,cblock_y,channel]
                
                    #Compute frame block
                    frame[chroma_block_pixel_upper_left_y:chroma_block_pixel_upper_left_y+chroma_block_height, chroma_block_pixel_upper_left_x:chroma_block_pixel_upper_left_x + chroma_block_width, channel] = \
                        ref_block + block_residuals

    return frame                 

def inter_frame_processing(height, width, height_chroma, width_chroma, frame, ref_frame, N, offset):
    if not ((N != 0) and (N & (N-1) == 0)):
        print("N is not power of 2")
        return -1
    
    motion_vectors_to_encode = {}
    residual_diffs = {}

    for cblock_x in tqdm(range(0,width//N)):
        for cblock_y in range(0,height//N):
            #for channel in range(0,3):
            
            #get current block
            cbx_pixel_upper_left = cblock_x * N
            cby_pixel_upper_left = cblock_y * N
            frame_block = frame.getMatrix(0)[cby_pixel_upper_left:cby_pixel_upper_left+N, cbx_pixel_upper_left:cbx_pixel_upper_left + N]

            #find best block for each channel
            diff = 0
            min_abs_diff = sys.maxsize
            best_block_x = 0
            best_block_y = 0
            best_diffs = []
            
            #Search for similar blocks in area of offset number of blocks around current block
            for bx in range(cblock_x-offset, cblock_x+offset+1):
                for by in range(cblock_y-offset, cblock_y+offset+1):

                    if bx >= 0 and bx < width//N and by>=0 and by<height//N:
                        #This is a valid block to compare with

                        #Get candidate reference block
                        bx_pixel_upper_left = bx * N
                        by_pixel_upper_left = by * N
                        ref_block = ref_frame.getMatrix(0)[by_pixel_upper_left:by_pixel_upper_left + N, bx_pixel_upper_left:bx_pixel_upper_left + N]

                        #Check if it is the best one
                        diff_block = frame_block - ref_block
                        abs_diff = np.absolute(diff_block).sum()
                        if abs_diff < min_abs_diff:
                            min_abs_diff = abs_diff
                            best_diff_block = diff_block
                            best_block_x = bx 
                            best_block_y = by

            #Compute motion vector for this channel
            motion_vector_x = best_block_x - cblock_x
            motion_vector_y = best_block_y - cblock_y
            
            #Compute residuals (matriz N*N com valores das diferenças residuais)
            #Its just best_diff_block
            
            #Write MV's and residuals to encode
            motion_vectors_to_encode[cblock_x,cblock_y] = (motion_vector_x,motion_vector_y)
            residual_diffs[cblock_x,cblock_y,0]  = best_diff_block


            chroma_height_ratio = height/height_chroma
            chroma_width_ratio = width/width_chroma

            chroma_block_height = int(N//chroma_height_ratio)
            chroma_block_width = int(N//chroma_width_ratio)
            for channel in range(1,3):
                cbx_pixel_upper_left = cblock_x * chroma_block_width
                cby_pixel_upper_left = cblock_y * chroma_block_height
                frame_block = frame.getMatrix(channel)[cby_pixel_upper_left:cby_pixel_upper_left+chroma_block_height, cbx_pixel_upper_left:cbx_pixel_upper_left + chroma_block_width]
                
                ref_bx_pixel_upper_left = best_block_x * chroma_block_width
                ref_by_pixel_upper_left = best_block_y * chroma_block_height
                ref_block = ref_frame.getMatrix(channel)[ref_by_pixel_upper_left:ref_by_pixel_upper_left + chroma_block_height, ref_bx_pixel_upper_left:ref_bx_pixel_upper_left + chroma_block_width]

                diff_block_ch = frame_block - ref_block
                residual_diffs[cblock_x,cblock_y,channel] = diff_block_ch 
                    
    return motion_vectors_to_encode, residual_diffs


def compare_blocks(frame, cbx, cby, ref_frame, bx, by, N, ch):
    #convert block coordenates to pixel coordinates
    #return -1, [1]*16
    cbx_pixel_upper_left = cbx * N
    cby_pixel_upper_left = cby * N
    
    bx_pixel_upper_left = bx * N
    by_pixel_upper_left = by * N
    
    '''
    overall_diff = 0
    diffs = []
    for x in range(0,N):
        for y in range(0,N):

            pixel_value = int( frame[cby_pixel_upper_left + y, cbx_pixel_upper_left + x, ch] )

            reference_pixel_value = int( ref_frame[by_pixel_upper_left + y, bx_pixel_upper_left + x, ch] ) 

            diff = pixel_value - reference_pixel_value 
            overall_diff += diff
            diffs += [diff]
    '''
    frame_block = frame[cby_pixel_upper_left:cby_pixel_upper_left+N, cbx_pixel_upper_left:cbx_pixel_upper_left + N, ch]
    ref_block = ref_frame[by_pixel_upper_left:by_pixel_upper_left + N, bx_pixel_upper_left:bx_pixel_upper_left + N, ch]

    overall_diff = np.absolute( frame_block - ref_block)

    # print(frame_block)
    # print(ref_block)
    # print(overall_diff)
    # input(overall_diff.sum())
    return overall_diff.sum(), ref_block


if __name__ == "__main__":
	main()