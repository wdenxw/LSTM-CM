# encoding=gbk
from torch import nn
import torch
from medianpool1d import MedianPool1d
from medianpool1d import MinPool2d


class BILSTM(nn.Module):
    """the model construction of dip
    the model construction of dip

    Args:
        None

    Returns:
        return the model output

    Raises:
        None
    """

    def __init__(self, input_size, output_size,seq_len, dropout):
        super(BILSTM, self).__init__()
        print("input_size:", input_size)
        print("output_size:", output_size)
        self.num_layers = 1

        self.seq_len = 5
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.seq_len, 512)  # 2 for bidirection
        self.relu = nn.ReLU()
        self.median=MedianPool1d()
        self.rnn = nn.LSTM(512, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.hidden_size = 512
        self.scalemin = 15
        self.avksize = 10
        self.ratio = 1
        self.calisize = seq_len
        self.min = MinPool2d(self.scalemin)
        self.fc3=nn.Linear(150, 150)
        self.fc4 = nn.Linear(512*2, 150)
        self.fc5 = nn.Linear(150, output_size)
        self.ave = nn.AvgPool2d(self.avksize, stride=None,
                                padding=int(self.avksize / 2),
                                ceil_mode=False, count_include_pad=True)
        self.fcts = nn.Linear(int(4), self.calisize)
        self.me = MedianPool1d(6)#


    def forward(self, inputs, **kwargs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        # sensor calibrate
        trans = inputs.permute(0, 2, 1)
        min = self.min(trans)
        yav = self.ave(min)
        lenn = yav.shape[0]
        if (yav.shape[-1] == 1):
            y1 = yav + torch.zeros((lenn, 1, 4)).cuda()
        else:
            self.fcts = nn.Linear(yav.shape[-1], self.calisize).cuda()
            y1 = yav
        y1 = self.fcts(y1)
        y1 = y1.permute(0, 2, 1)
        y1 = inputs - y1
        #bi-lstm module
        y1 = inputs[:, -5:, :]
        inputs = y1.permute(0,2,1)
        y1 = inputs
        y3 = self.fc1(y1)
        h0 = torch.zeros(self.num_layers * 2, inputs.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers * 2, inputs.size(0), self.hidden_size).cuda()
        y5, _ = self.rnn(y3, (h0, c0))
        y6 = self.fc3(self.fc4(y5))
        y7=self.fc5(y6)
        y8 =y7[:, -1, :]
        out = self.me(y8 )
        return out