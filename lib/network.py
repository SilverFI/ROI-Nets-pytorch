import torch
from torch import nn
from ROILayer import ROILayer

class ROINet(nn.Module):
    def __init__(self,num_classes=12,seq_num=24,cuda_num=0):
        super(ROINet, self).__init__()
        self.vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512]
        self.features = make_layers(self.vgg_cfg)    #vgg backbone
        self.roi = ROILayer(cuda_num)
        self.fc1 = nn.Sequential(nn.Linear(in_features=3000,out_features=2048),
                                 nn.ReLU())
        self.lstm = nn.LSTM(input_size=2048,hidden_size=512,num_layers=1,batch_first=True)
        self.fc2 = nn.Sequential(nn.Linear(in_features=512,out_features=num_classes),
                                 nn.Sigmoid())
        self.seq_num = seq_num

    def forward(self,x,y):
        vgg_features = self.features(x)
        roi = self.roi(vgg_features,y)
        roi_fc = self.fc1(roi)
        roi_fc = roi_fc.view(-1,self.seq_num,2048)
        lstm_features,_ = self.lstm(roi_fc)
        output = self.fc2(lstm_features.contiguous().view(-1,512))
        return output

    # loss function
    @staticmethod
    def multi_label_ACE(outputs, y_labels):
        batch_size,class_size = outputs.size()
        loss_buff = 0
        for i in range(class_size):
            target = y_labels[:, i]
            output = outputs[:, i]
            loss_au = torch.sum(-(target * torch.log((output + 0.05) / 1.05) + (1.0 - target) * torch.log((1.05 - output) / 1.05)))
            loss_buff += loss_au
        return loss_buff / (class_size * batch_size)

    # inputs:[prediction,label,thresh]  outputs: [{TP,FP,TN,FN}*class_num] for class_num AUs
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def statistics(pred, y, thresh):
        batch_size = pred.size(0)
        class_nb = pred.size(1)

        pred = pred > thresh
        pred = pred.long()
        pred[pred == 0] = -1
        y[y == 0] = -1

        statistics_list = []
        for j in range(class_nb):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            for i in range(batch_size):
                if pred[i][j] == 1:
                    if y[i][j] == 1:
                        TP += 1
                    elif y[i][j] == -1:
                        FP += 1
                    else:
                        assert False
                elif pred[i][j] == -1:
                    if y[i][j] == 1:
                        FN += 1
                    elif y[i][j] == -1:
                        TN += 1
                    else:
                        assert False
                else:
                    assert False
            statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
        return statistics_list

    # inputs: [{TP,FP,TN,FN}*class_num] for class_num AUs  outputs: mean F1 scores and lists
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def calc_f1_score(statistics_list):
        f1_score_list = []

        for i in range(len(statistics_list)):
            TP = statistics_list[i]['TP']
            FP = statistics_list[i]['FP']
            FN = statistics_list[i]['FN']

            precise = TP / (TP + FP + 1e-20)
            recall = TP / (TP + FN + 1e-20)
            f1_score = 2 * precise * recall / (precise + recall + 1e-20)
            f1_score_list.append(f1_score)
        mean_f1_score = sum(f1_score_list) / len(f1_score_list)

        return mean_f1_score, f1_score_list

    # update statistics list
    # ref: https://github.com/AlexHex7/DRML_pytorch
    @staticmethod
    def update_statistics_list(old_list, new_list):
        if not old_list:
            return new_list

        assert len(old_list) == len(new_list)

        for i in range(len(old_list)):
            old_list[i]['TP'] += new_list[i]['TP']
            old_list[i]['FP'] += new_list[i]['FP']
            old_list[i]['TN'] += new_list[i]['TN']
            old_list[i]['FN'] += new_list[i]['FN']

        return old_list

# helper for making vgg net
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)