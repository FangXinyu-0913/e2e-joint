from gettext import Catalog
import os
import struct
import math
import cv2
import numpy as np

# def getDVSeventsDavis(file, numEvents=5e4, startTime=0):
def getDVSeventsDavis(file, output_file, numEvents=5e4, startTime=0):
    """ DESCRIPTION: This function reads a given aedat file and converts it into four lists indicating 
                     timestamps, x-coordinates, y-coordinates and polarities of the event stream. 
    
    Args:
        file: the path of the file to be read, including extension (str).
        numEvents: the maximum number of events allowed to be read (int, default value=1e10).
        startTime: the start event timestamp (in microseconds) where the conversion process begins (int, default value=0).

    Return:
        ts: list of timestamps in microseconds.
        x: list of x-coordinates in pixels.
        y: list of y-coordinates in pixels.
        pol: list of polarities (0: on -> off, 1: off -> on).       
    """
    # print('\ngetDVSeventsDavis function called \n')
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY
   
    # print('Reading in at most', str(numEvents))
    triggerevent = int('400', 16)
    polmask = int('800', 16)
    xmask = int('003FF000', 16)
    ymask = int('7FC00000', 16)
    typemask = int('80000000', 16)
    typedvs = int('00', 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []
    pol = []
    events = np.empty(shape=(0,4))
    numeventsread = 0
    numeventspack = 0
    
    length = 0
    aerdatafh = open(file, 'rb')
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    # print("file size", length)

    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    events_num = 0
    no = 1
    while p < length:
        ad, tm = struct.unpack_from('>II', tmp)
        ad = abs(ad)
        if tm >= startTime:
            if (ad & typemask) == typedvs:
                xo = sizeX - 1 - float((ad & xmask) >> xshift)
                yo = float((ad & ymask) >> yshift)
                polo = 1 - float((ad & polmask) >> polshift)
                if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                    # tmp_list = []
                    # tmp_list.append(xo)
                    # tmp_list.append(yo)
                    # tmp_list.append(tm)
                    # tmp_list.append(polo)
                    events_num +=1
                    np_array= np.array((xo,yo,(tm - startTime),polo))
                    events = np.row_stack((events,np_array.astype(np.float32)))
                    if(events_num >= 2e5):
                        np.save(output_file + str(no),events)
                        print(output_file + str(no)+".npy is stored!")
                        events = np.empty(shape=(0,4))
                        no += 1
                        events_num = 0
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1

    # print('Total number of events read =', numeventsread)
    # print('Total number of DVS events returned =', len(ts))

    # return ts, x, y, pol
    # print(events)
    # print(np.min(events[:,2], axis=0))
    # minnum = np.min(events[:,2], axis=0)
    # events[:,2] -= minnum
    # print(file)
    # print(events)
    return events

if __name__ == '__main__':
    catalogues = ['train','validation','test']
    print(os.getcwd())
    kinds = ['arm-crossing','falling-down','get-up','kicking','picking-up','sit-down','throwing','turning-around','tying-shoes','walking','waving']
    # train
    for kind in kinds:
        input_route = 'train/' + kind + '/'
        for i in range(5):
            for j in range(3):
                input_file = os.getcwd()+ '/FallDown_ActionRecognition/' + input_route + 'f_' + str(i+1) + '.' + str(j+1) + '.aedat'
                output_file = os.getcwd()+ '/FallDown_ActionRecognition_new/' + input_route + 'f_' + str(i+1) + '.' + str(j+1) + '.'
                print(input_file)
                events = getDVSeventsDavis(input_file, output_file)
                # T,X,Y,P = getDVSeventsDavis(input_file)

    # validation
    for kind in kinds:
        input_route = 'validation/' + kind + '/'
        for i in range(3):
            for j in range(3):
                input_file = os.getcwd()+ '/FallDown_ActionRecognition/' + input_route + 'f_' + str(i+1) + '.' + str(j+1) + '.aedat'
                output_file = os.getcwd()+ '/FallDown_ActionRecognition_new/' + input_route + 'f_' + str(i+1) + '.' + str(j+1) + '.'
                print(input_file)
                events = getDVSeventsDavis(input_file, output_file)

    # test
    for kind in kinds:
        input_route = 'test/' + kind + '/'
        for i in range(2):
            for j in range(3):
                input_file = os.getcwd()+ '/FallDown_ActionRecognition/' + input_route + 'f_' + str(i+1) + '.' + str(j+1) + '.aedat'
                output_file = os.getcwd()+ '/FallDown_ActionRecognition_new/' + input_route + 'f_' + str(i+1) + '.' + str(j+1) + '.'
                print(input_file)
                events = getDVSeventsDavis(input_file, output_file)
