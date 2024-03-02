import numpy as np
import torch
import random
dataOnePath = '/home/wdenxw/pp/whd/'
Columy = [
            'RightShoulderZ', 'RightShoulderX', 'RightShoulderY',
            'LeftShoulderZ', 'LeftShoulderX', 'LeftShoulderY',
            'RightUpLegZ', 'RightUpLegX', 'RightUpLegY', 'RightLegZ',
            'RightLegY', 'RightLegX', 'LeftUpLegZ', 'LeftUpLegX', 'LeftUpLegY', 'LeftLegZ',
            'LeftLegY', 'LeftLegX', 'SpineZ', 'SpineX', 'SpineY', 'Spine1Z', 'Spine1X', 'Spine1Y',
            'Spine2Z', 'Spine2X', 'Spine2Y', 'RightArmZ', 'RightArmX',
            'RightArmY', 'RightForeArmZ', 'RightForeArmX', 'RightForeArmY',
            'LeftArmZ', 'LeftArmX', 'LeftArmY', 'LeftForeArmZ', 'LeftForeArmX', 'LeftForeArmY'
        ]
def ToNormalize(inputArr):
    min = np.min(inputArr)
    max = np.max(inputArr)
    yy = max - min
    lastlist = []
    for i in range(len(inputArr)):
        inputArr[i] = (inputArr[i] - min)
        lastlist.append(inputArr[i] / yy)
    return np.array(lastlist)
def channelNom(inputArr,computeminmax,inputmin,inputmax):
    channelmin=[]
    channelmax=[]
    for i in range(inputArr.shape[1]):
        if(computeminmax):
            #channel normalization
            min = torch.min(inputArr[:,i])
            max = torch.max(inputArr[:,i])
            channelmin.append(min.cpu().numpy())
            channelmax.append(max.cpu().numpy())
        else:
            min=torch.from_numpy(inputmin[i])
            max=torch.from_numpy(inputmax[i])
        yy=max-min
        for j in range(inputArr.shape[0]):
            inputArr[j,i] = (inputArr[j,i] - min)/yy
    return inputArr,channelmin,channelmax
def oneseqNom(inputArr,min,max):
    yy = max - min
    for i in range(len(inputArr)):
        inputArr[i] = (inputArr[i] - min)/yy
    return np.array(inputArr)
def seqNormalize(inputArr):# normalize the training data
    wholemin = np.min(inputArr)
    wholemax = np.max(inputArr)
    for i in range(inputArr.shape[2]):
        # min=np.min(inputArr[:,:,i])
        # max=np.max(inputArr[:,:,i])
        for j in range(inputArr.shape[1]):
            inputArr[:,j,i]=oneseqNom(inputArr[:,j,i],wholemin,wholemax)
    return np.array(inputArr)
def DoNormalize(inputnpy):
    for i in range(inputnpy.shape[2]):
        inputnpy[:, :, i] = ToNormalize(inputnpy[:, :, i])
    return inputnpy
def transefcnData(X_train,X_test,X_valid,isseq):
    if(isseq):
        X_train=X_train.reshape(X_train.shape[0],-1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        X_valid = X_valid.reshape(X_valid.shape[0], -1)
    else:
        X_train=X_train[:, -1, :]
        X_test=X_test[:, -1, :]
        X_valid = X_valid[:, -1, :]
    return X_train,X_test,X_valid

def transdataraw(issingle,seq,shuffle):
    X_trainR = np.load(dataOnePath + 'X_train200.npy')
    Y_trainR = np.load(dataOnePath + 'Y_train200.npy')[:,-1,:]
    X_testR = np.load(dataOnePath + 'X_test200.npy')
    Y_testR = np.load(dataOnePath + 'Y_test200.npy')[:,-1,:]
    Y_validR = np.load(dataOnePath + 'Y_vali200.npy')[:,-1,:]
    X_validR = np.load(dataOnePath + 'X_vali200.npy')
    X_trainR=X_trainR[:,X_trainR.shape[1]-seq:,:]
    X_testR = X_testR[:, X_testR.shape[1]-seq: , :]
    X_validR = X_validR[:, X_validR.shape[1]-seq:, :]
    if (issingle):
        X_trainR,X_testR,X_validR=transefcnData(X_trainR,X_testR,X_validR,False)
        X_trainR = ToNormalize(X_trainR)
        X_testR = ToNormalize(X_testR)
        X_validR = ToNormalize(X_validR)
    else:
        X_trainR=seqNormalize(X_trainR)
        X_testR=seqNormalize(X_testR)
        X_validR=seqNormalize(X_validR)
    if (shuffle):
        cc = list(zip(X_trainR, Y_trainR))
        random.shuffle(cc)
        X_trainR, Y_trainR = zip(*cc)
    Y_train = torch.from_numpy(np.array(Y_trainR)).float().cuda()
    X_train = torch.from_numpy(np.array(X_trainR)).float().cuda()
    X_test = torch.from_numpy(X_testR).float().cuda()
    Y_test = torch.from_numpy(Y_testR).float().cuda()
    X_valid = torch.from_numpy(X_validR).float().cuda()
    Y_valid = torch.from_numpy(Y_validR).float().cuda()

    print("dataset prepared")
    return X_train,Y_train,X_test,Y_test,X_valid,Y_valid