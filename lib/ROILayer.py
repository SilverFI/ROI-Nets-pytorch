import torch
from torch import nn


class ROILayer(nn.Module):
    def __init__(self,cuda_num=0):
        super(ROILayer, self).__init__()
        self.roi_layer = dict()
        self.cuda_num = cuda_num
        for i in range(20):
            module_name = 'roi_conv_%d' % (i)
            self.roi_layer[module_name] = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                nn.ReLU()
            )  # upsample and convlution
            self.add_module(name=module_name, module=self.roi_layer[module_name])

            module_name = 'roi_fc_%d' % (i)
            self.roi_layer[module_name] = nn.Sequential(
                nn.Linear(in_features=512*6*6,out_features=150),
                nn.ReLU()
            ) # fc
            self.add_module(name=module_name, module=self.roi_layer[module_name])


    def process_landmarks(self,landmarks):
        # get roi region according to landmarks
        batch_size,_,_ = landmarks.size()
        region_array = torch.zeros(batch_size,10,4)
        if torch.cuda.is_available():
            region_array = region_array.cuda(self.cuda_num)
        arr2d = torch.transpose(landmarks,1,2)
        ruler = torch.abs(arr2d[:,0, 39] - arr2d[:,0, 42])
        region_array[:,0,:] = torch.stack((arr2d[:,0, 21], arr2d[:,1, 21] - ruler / 2, arr2d[:,0, 22], arr2d[:,1, 22] - ruler / 2),1)     #  1:  AU1
        region_array[:,1,:] = torch.stack((arr2d[:,0, 18], arr2d[:,1, 18] - ruler / 3, arr2d[:,0, 25], arr2d[:,1, 25] - ruler / 3),1)     #  2:	 AU2
        region_array[:,2,:] = torch.stack((arr2d[:,0, 19], arr2d[:,1, 19] + ruler / 3, arr2d[:,0, 24], arr2d[:,1, 24] + ruler / 3),1)     #  3:  AU4
        region_array[:,3,:] = torch.stack((arr2d[:,0, 41], arr2d[:,1, 41] + ruler, arr2d[:,0, 46], arr2d[:,1, 46] + ruler),1)             #  4:  AU6
        region_array[:,4,:] = torch.stack((arr2d[:,0, 38], arr2d[:,1, 38], arr2d[:,0, 43], arr2d[:,1, 43]),1)                             #  5:  AU7
        region_array[:,5,:] = torch.stack((arr2d[:,0, 49], arr2d[:,1, 49], arr2d[:,0, 53], arr2d[:,1, 53]),1)                             #  6:  AU10
        region_array[:,6,:] = torch.stack((arr2d[:,0, 48], arr2d[:,1, 48], arr2d[:,0, 54], arr2d[:,1, 54]),1)                             #  7:  AU12 AU14 AU15
        region_array[:,7,:] = torch.stack((arr2d[:,0, 51], arr2d[:,1, 51], arr2d[:,0, 57], arr2d[:,1, 57]),1)                             #  8:  AU23 24
        region_array[:,8,:] = torch.stack((arr2d[:,0, 61], arr2d[:,1, 61], arr2d[:,0, 63], arr2d[:,1, 63]),1)                             #  9:  AU10
        region_array[:,9,:] = torch.stack((arr2d[:,0, 56], arr2d[:,1, 56] + ruler / 2, arr2d[:,0, 58], arr2d[:,1, 58] + ruler / 2),1)     # 10:  AU17
        return region_array/16.0


    def forward(self, x,y):
        batch_size, channels, height, width = x.size()
        new_input = torch.zeros(batch_size,channels*20,3,3)
        if torch.cuda.is_available():
            new_input = new_input.cuda(self.cuda_num)
        pos_para = self.process_landmarks(y).int()
        pos_para = torch.clamp(pos_para,min=1,max=12)

        # crop using roi
        for ii in range(batch_size):
            for j in range(10):
                p = pos_para[ii, j, :]
                new_input[ii,512*2*j:512*(2*j+1),:,:] = x[ii,:,p[1]-1:p[1]+2,p[0]-1:p[0]+2]
                new_input[ii,512*(2*j+1):512*(2*j+2),:,:] = x[ii,:,p[3]-1:p[3]+2,p[2]-1:p[2]+2]

        # process roi region
        for i in range(20):
            region = new_input[:,i*512:(i+1)*512,:,:]
            module_name = 'roi_conv_%d' % (i)
            conv_res = self.roi_layer[module_name](region)
            module_name = 'roi_fc_%d' % (i)
            if i==0:
                fc_res = self.roi_layer[module_name](conv_res.view(batch_size,-1))
            else:
                fc_res = torch.cat((fc_res,self.roi_layer[module_name](conv_res.view(batch_size,-1))),1)
        return fc_res