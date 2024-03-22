import numpy as np
import sys, os, gzip, time
from msTools import *

msInFileName, demogParamFileName, outFileName = sys.argv[1:]

num_test=10000 # only operate on the first 10000 for scramble test

def combineHapsAndSnpDists(haplotypeMatrices, distsBetweenSnpVectors, 
                           maxSnps, maxDistBtwnSnps,
                           scramble_mode=None):
    sampleSize = len(haplotypeMatrices[0])
    # print sampleSize # 50
    # create an empty arr with size: 10000, 51, 730
    inputMatrices = np.empty((len(haplotypeMatrices[:num_test]), sampleSize+1, maxSnps+1), dtype='uint8')

    assert len(haplotypeMatrices) == len(distsBetweenSnpVectors) # num data sets
    # print(np.array(haplotypeMatrices[0]).shape) #  haplotypeMatrices[i] is a data set with shape (# chrs, # snps)
    # print(np.array(haplotypeMatrices[0]))
    
    # iterate through each data set of the first 10,000
    # generate scramble data set before padding
    # at this point haplotypeMatrices is not padded so it is ragged,
    # also technically a list
    
    for i in range(len(haplotypeMatrices[:num_test])):
        # different scramble modes
        if scramble_mode == "col" or scramble_mode == "colI": 
            # transpose the haplotype matrix
            arr = np.array(haplotypeMatrices[i]).T
            # shuffle the transposed rows, which were SNP columns originally
            np.random.shuffle(arr) 
            
            # if also transpose entries within columns (long-range LD)
            if scramble_mode == "colI":
                for j in range(arr.shape[0]): # iterate through each row, which are the snp columns 
                    np.random.shuffle(arr[j]) # shuffle within each rows, which are snp columns
            
            # retranspose and convert back to list after scrambling
            haplotypeMatrices[i] = arr.T.tolist()
            
        if scramble_mode == "free":
            arr = np.array(haplotypeMatrices[i])
            d1, d2 = arr.shape

            # reshape to 1D arr, then shuffle, which is equivalent to free shuffle
            arr = arr.reshape(d1 * d2)
            np.random.shuffle(arr)
            arr = arr.reshape(d1, d2)
            haplotypeMatrices[i] = arr.tolist() # convert back to list after scrambling

        if scramble_mode == "right" or scramble_mode == "left":
            arr = np.array(haplotypeMatrices[i]).astype('int')
            row, col = arr.shape
            num_ones = np.count_nonzero(arr)

            if scramble_mode == "left": # SNPs will be pushed to left-most columns
                arr_parted = np.concatenate([np.ones(num_ones), np.zeros(row * col - num_ones)])
            
            if scramble_mode == "right": # SNPs will be pushed to right-most columns
                arr_parted = np.concatenate([np.zeros(row * col - num_ones), np.ones(num_ones)])
            
            arr_parted = arr_parted.astype('int').reshape((col, row)).T
            haplotypeMatrices[i] = arr_parted.astype('str').tolist() # convert back to list and str type after scrambling
        

    for i in range(len(haplotypeMatrices[:num_test])): 
    # for each data set, do padding for the tensor and distance vector
        unpaddedLen = len(distsBetweenSnpVectors[i])
        assert unpaddedLen == len(haplotypeMatrices[i][0]) + 1
        padLen = maxSnps - unpaddedLen
        # distsBetweenSnpVectors includes distances to edges of chromosome
        k = 0 # iterator
        
        # this is working on just the position vector (hence the [0])
        while k < len(distsBetweenSnpVectors[i]): # num of SNPs without padding
            inputMatrices[i][0][k] = int(round(255*distsBetweenSnpVectors[i][k]/maxDistBtwnSnps))
            k += 1
            
        while k < maxSnps+1: # this is padding with 0 for the position vector
            inputMatrices[i][0][k] = 0
            k += 1
        
        # I think the rest is for the rest of the tensor matrix (hence the j+1)
        for j in range(sampleSize): # sampleSize = num chromosomes = 50
            k = 0
            while k < len(haplotypeMatrices[i][j]):
                inputMatrices[i][j+1][k] = 255*int(haplotypeMatrices[i][j][k])
                k += 1
            while k < maxSnps+1: # this is padding again
                inputMatrices[i][j+1][k] = 0
                k += 1

    # print(inputMatrices.shape) # (10000, 51, 729)
    return inputMatrices

def getDistancesBetweenSnps(positionVectors):
    distVectors = []
    max_dist = 0
    for i in range(len(positionVectors)):
        currVec = []
        prevPos = 0.0
        for j in range(len(positionVectors[i])):
            currVec.append(positionVectors[i][j]-prevPos)
            prevPos = positionVectors[i][j]
            
            # if i == 0: # print an example
                # print currVec
                # print prevPos
            
        currVec.append(1.0-prevPos)
        distVectors.append(currVec)
        
        # get max distance
        curr_max_dist = max(currVec)
        # print curr_max_dist
        # update if larger
        if curr_max_dist > max_dist:
            max_dist = curr_max_dist
            
    # print max_dist
    return distVectors, max_dist

def readDemogParams(demogParamPath):
    params = []
    first = True
    with open(demogParamPath) as demogParamFile:
        for line in demogParamFile:
            if first:
                first = False
            else:
                params.append([float(x) for x in line.strip().split()])
    return params

haplotypeMatrices, positionVectors, demogParams = [], [], []
haplotypeMatrices, positionVectors, maxSnps = msOutToHaplotypeMatrix(msInFileName)
demogParams = readDemogParams(demogParamFileName)
assert len(positionVectors) == len(haplotypeMatrices)
distsBetweenSnpVectors, maxDistBtwnSnps = getDistancesBetweenSnps(positionVectors)

print("maxSnps: " + str(maxSnps))
y = np.array(demogParams[:num_test])
print("y.shape: " + str(y.shape))

for mode in ["col", "colI", "free", "right", "left"]:
    print("Scramble mode: " + mode)
    X = np.array(combineHapsAndSnpDists(haplotypeMatrices, distsBetweenSnpVectors, 
                                        maxSnps, maxDistBtwnSnps, scramble_mode=mode))
    print("X.shape: " + str(X.shape))
    outFileNameMode = outFileName + "_" + mode
    np.savez_compressed(outFileNameMode, X=X, y=y)
