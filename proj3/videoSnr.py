import numpy as np
import sys
from math import log10

from VideoCaptureYUV import VideoCaptureYUV

#sys.path.append('..')

def main():
    if(len(sys.argv) <= 2):
        print("USAGE: python3 {} inputFile1 inputFile2".format(sys.argv[0]))
        return
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    snrValue = snrAll(file1, file2)
    #snrValue = snrY(file1, file2)

    print(snrValue)

def snrY(file1, file2):
    cap1 = VideoCaptureYUV(file1)
    cap2 = VideoCaptureYUV(file2)

    total_sum = 0
    numberOfFrames = 0
    while True:
        ret1, frame1 = cap1.read_raw()
        ret2, frame2 = cap2.read_raw()
        if ret1 and ret2:
            numberOfFrames += 1
            y1 = frame1.getMatrix(0)
            y2 = frame2.getMatrix(0)

            diffs = abs(y1 - y2)

            total_sum += np.sum(diffs * diffs)

        else:
            break
    mse = total_sum / (cap1.width * cap1.height * numberOfFrames)

    psnr = 10.0*log10((255*255)/mse)
    return psnr

def snrAll(file1, file2):
    cap1 = VideoCaptureYUV(file1)
    cap2 = VideoCaptureYUV(file2)

    total_sum = 0
    numberOfFrames = 0
    while True:
        ret1, frame1 = cap1.read_raw()
        ret2, frame2 = cap2.read_raw()
        if ret1 and ret2:
            numberOfFrames += 1

            y1 = frame1.getMatrix(0)
            y2 = frame2.getMatrix(0)
            u1 = frame1.getMatrix(1)
            u2 = frame2.getMatrix(1)
            v1 = frame1.getMatrix(2)
            v2 = frame2.getMatrix(2)

            diffsY = abs(y1 - y2)
            diffsU = abs(u1 - u2)
            diffsV = abs(v1 - v2)
            total_sum += np.sum(diffsY * diffsY)
            total_sum += np.sum(diffsU * diffsU)
            total_sum += np.sum(diffsV * diffsV)

        else:
            break
    mse = total_sum / ( (cap1.width * cap1.height + 2 * cap1.width_chroma * cap1.height_chroma) * numberOfFrames)

    psnr = 10.0*log10((255*255)/mse)
    return psnr

if __name__ == "__main__":
    main()
