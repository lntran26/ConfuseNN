import numpy as np
import sys, os, gzip, time
from msTools import *

msInFileName, demogParamFileName, outFileName = sys.argv[1:]

def combineHapsAndSnpDists(haplotypeMatrices, distsBetweenSnpVectors, maxSnps, maxDistBtwnSnps):
    sampleSize = len(haplotypeMatrices[0])
    # print sampleSize # 50
    inputMatrices = np.empty((len(haplotypeMatrices), sampleSize+1, maxSnps+1), dtype='uint8')
    assert len(haplotypeMatrices) == len(distsBetweenSnpVectors) # num data sets
    for i in range(len(haplotypeMatrices)): # for each data set
        unpaddedLen = len(distsBetweenSnpVectors[i])
        assert unpaddedLen == len(haplotypeMatrices[i][0]) + 1
        padLen = maxSnps - unpaddedLen
        # distsBetweenSnpVectors includes distances to edges of chromosome
        k = 0
        # print len(distsBetweenSnpVectors[i]) # this equals unpaddedLen for each data set
        
        # I think this is working on just the position vector (hence the [0])
        while k < len(distsBetweenSnpVectors[i]):
            inputMatrices[i][0][k] = int(round(255*distsBetweenSnpVectors[i][k]/maxDistBtwnSnps))
            k += 1
        while k < maxSnps+1: # I think this is padding with 0
            inputMatrices[i][0][k] = 0
            k += 1
        
        # I think the rest is for the rest of the tensor matrix (hence the j+1)
        for j in range(sampleSize):
            k = 0
            while k < len(haplotypeMatrices[i][j]):
                inputMatrices[i][j+1][k] = 255*int(haplotypeMatrices[i][j][k])
                k += 1
            while k < maxSnps+1: # this is padding again
                inputMatrices[i][j+1][k] = 0
                k += 1
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

X = np.array(combineHapsAndSnpDists(haplotypeMatrices, distsBetweenSnpVectors, maxSnps, maxDistBtwnSnps))
y = np.array(demogParams)

np.savez_compressed(outFileName, X=X, y=y)
