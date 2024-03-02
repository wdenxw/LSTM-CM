# encoding=gbk
# !/usr/bin/python3
"""train the model

Args:
    epoch: current epoch
Returns:
    print the loss and tracking results

Raises:
    None
"""
import numpy as np
import os
import filter as OF
import dataloder as loder
import datetime as dt
from settings import settings
import torch
import torch.optim as optim
import torch.nn.functional as F

class intialize(object):
    def __init__(self):
        Columx = ['ResRightArm1', 'ResLeftArm1','ResRightLeg1', 'ResLeftLeg1']
        self.CurrentModelSensor = 'bilstm'
        self.inputbatch=False
        self.filter='None'#median,None#
        self.seq_len = 5
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.outputpath='output/'
        self.Columy = [
            'RightShoulderZ', 'RightShoulderX', 'RightShoulderY',
            'LeftShoulderZ', 'LeftShoulderX', 'LeftShoulderY',
            'RightUpLegZ', 'RightUpLegX', 'RightUpLegY', 'RightLegZ',
            'RightLegY', 'RightLegX', 'LeftUpLegZ', 'LeftUpLegX', 'LeftUpLegY', 'LeftLegZ',
            'LeftLegY', 'LeftLegX', 'SpineZ', 'SpineX', 'SpineY', 'Spine1Z', 'Spine1X', 'Spine1Y',
            'Spine2Z', 'Spine2X', 'Spine2Y', 'RightArmZ', 'RightArmX',
            'RightArmY', 'RightForeArmZ', 'RightForeArmX', 'RightForeArmY',
            'LeftArmZ', 'LeftArmX', 'LeftArmY', 'LeftForeArmZ', 'LeftForeArmX', 'LeftForeArmY'
        ]

        st=settings()
        self.args, unknown = st.parser.parse_known_args()
        self.batch_size = st.args.batch_size
        self.input_channels = len(Columx)  # input_size
        self.epochs = st.args.epochs
        self.validlist=[]
        self.earlystopping=False
        self.dropout= st.args.dropout
        self.clip=st.args.clip
        self.output_interval=st.args.output_interval
        self.steps = 0
        self.n_classes = len(self.Columy)  # output_size



    def train(self,epoch):
        """train the model
        train the dip model

        Args:
            epoch: current epoch
        Returns:
            print the loss

        Raises:
            None
        """
        self.model.train()
        batch_idx = 1
        total_loss = 0
        train_losses = []

        for i in range(0, self.X_train.size(0), self.batch_size):
            if i + self.batch_size > self.X_train.size(0):
                continue
            else:
                x, y = self.X_train[i:(i + self.batch_size)], self.Y_train[i:(i + self.batch_size)]
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            if self.args.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            batch_idx += 1
            total_loss += loss.item()
            train_losses.append(loss.item())
            if batch_idx % self.args.log_interval == 0:
                cur_loss = total_loss / self.args.log_interval

                processed = min(i + self.batch_size, self.X_train.size(0))
                print('Train Epoch: {:2d} [{:6d}/{:6d} '
                      '({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, processed, self.X_train.size(0),
                    100. * processed / self.X_train.size(0), self.lr, cur_loss))
                total_loss = 0

    def batchinput(self,Xdata):
        count = int(Xdata.shape[0] / 64)
        for i in range(count):
            if (i == 0):
                raw = self.model(Xdata[i * 64:(i + 1) * 64, :])
            else:
                new = self.model(Xdata[i * 64:(i + 1) * 64, :])
                outputTorch = torch.vstack((raw, new))
                raw = outputTorch
        return outputTorch
    def computeErr(self,Xdata, Ydata, name,inputbatch,channelminTr=[0],channelmaxTr=[0]):
        self.joint = [
            'RightShoulder', 'LeftShoulder','RightUpLeg',  'RightLeg',
             'LeftUpLeg',  'LeftLeg','Spine','Spine1',
            'Spine2', 'RightArm','RightForeArm', 'LeftArm', 'LeftForeArm' ]
        endtime1 = dt.datetime.now()
        print("consuming time:", (endtime1 - self.starttime1) / len(self.X_test))
        Ydata=Ydata.cpu().numpy()
        if (inputbatch):
            outputTorch=self.batchinput(Xdata)
        else:
            start_time2=dt.datetime.now()
            outputTorch = self.model(Xdata)
            now_time2 = dt.datetime.now()
            if(name=='test'):
                print("single data consuming time:", (now_time2 - start_time2) / len(self.X_test))
        output = outputTorch.cpu().numpy()
        self.computeJitter(output)
        if(self.filter=='median'):
            output=OF.median_filter(output)
        elif(self.filter=='None'):
            output = output

        print("output:",output.shape,"ydata:",Ydata.shape)
        error = abs(output - Ydata)
        print("The result of " +self.CurrentModelSensor+ "in" + name + "set" )
        print("Average error",np.mean(error),"Variance of error:",np.var(error))

        print("after filter output:", output.shape, "ydata:", Ydata.shape)
        output = OF.median_filter(output)
        error = abs(output - Ydata)
        print("Results in " + name + "set:", )
        print("Avereage error:", np.mean(error), "Variance of error:", np.var(error))

        if(name=='validate'):#
            self.validlist.append(np.mean(error))
    def computeJitter(self,inputPred):
        npours = inputPred
        n1 = npours[1:][:]
        n2 = npours[0:-1][:]
        movingDis = abs(n2 - n1)
        jitter = 6 * movingDis / (0.05 ** 3) / 10000
        averagePause = np.sum(jitter, axis=0) / jitter.shape[0]
        average = np.sum(averagePause) / jitter.shape[1]
        print("jitter:",average)

    def evaluate(self,channelminTr=[0],channelmaxTr=[0]):
        # output the training result
        self.model.eval()
        with torch.no_grad():
             self.computeErr(self.X_test, self.Y_test, 'test',self.inputbatch,channelminTr,channelmaxTr)
             self.computeErr(self.X_valid, self.Y_valid, 'validate',self.inputbatch,channelminTr,channelmaxTr)

    def usemodel(self):
        from BI_LSTMmodel import BILSTM
        self.modelPath = 'netmodel/bilstm.pth'
        self.model = BILSTM(self.input_channels, self.n_classes, self.seq_len,
                            dropout=self.args.dropout)
        self.X_train, self.Y_train, self.X_test, \
        self.Y_test, self.X_valid, self.Y_valid = loder.transdataraw(False, self.seq_len,False)
if __name__ == "__main__":
    mm=intialize()
    # whether only regress joint where sensors are located
    # Load the data and define the model save path
    mm.usemodel()
    mm.model.cuda()
    print("model success cuda")
    mm.lr = mm.args.lr
    mm.optimizer = getattr(optim, mm.args.optim)(mm.model.parameters(), lr=mm.lr)
    mm.starttime1 = dt.datetime.now()
    for epoch in range(1, mm.epochs + 1):
        mm.train(epoch)
        mm.model.eval()
        if epoch%mm.output_interval==0:
            mm.evaluate()
        if epoch % 10 == 0:
            mm.lr /= 10
        for param_group in mm.optimizer.param_groups:
            param_group['lr'] = mm.lr
            # early stoppling
        if (len(mm.validlist) > 11):
            if (min(mm.validlist[0:-(int(10 / mm.output_interval))])-min(mm.validlist)<0.1 ):
                mm.earlystopping = True
                print("stopped!!!!!!!")#
            # if the verification set error does not change in another 100 generations, exit the loop
            if (mm.earlystopping):
                endtime1 = dt.datetime.now()
                print("training consuming time:", (endtime1 - mm.starttime1))
                torch.save(mm.model.state_dict(), mm.modelPath)
                break