import torch
import os
import random
from torchvision import transforms
from PIL import Image
from helper import get_im_dict,get_label_landmark,train_test_split
from network import ROINet
import logging,time,shutil


save_dir = './models/'                                                                                               # directory to save models and log
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
else:
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)
log_path = save_dir+'/run.log'                                                                                        # log path
image_dir = "your_image_dir"          # images dir
class_number = 12                                                                                                     # number of AUs
cuda_num = 0                                                                                                          # only support signle gpu card
person_batch = 4                                                                                                      # how many sequences in one batch
seq_num = 24                                                                                                          # sequence length
au_thresh = 0.5                                                                                                       # AU thresh, default:0.5
epochs = 10                                                                                                           # number of epochs 
batches = 2000                                                                                                        # how many iterations in one epoch
print_every = 40                                                                                                      # print info every

print('loading...')
im_dic = get_im_dict()                                                                                                # im_dic: {'F001_T1':[2440,2441,...],'F002_T2':[],...}
labels, landmarks = get_label_landmark()                                                                              # labels: {im_name:[labels]} landmarks: {im_name:[landmark]}
train_person,test_person = train_test_split()                                                                         # train/test_person = ['F001_T1','F001_T2','F001_T3','F001_T4','F001_T5','F001_T6','F001_T7','F001_T8'...]

# logging
logging.basicConfig(level=logging.INFO,
                    format='(%(asctime)s %(levelname)s) %(message)s',
                    datefmt='%d %b %H:%M:%S',
                    filename=log_path,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('(%(levelname)s) %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# load pretrained model and fix previous layers
pretrained_model = torch.load('../data/vgg16.pth')                                                                    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
net = ROINet(num_classes=class_number,seq_num=seq_num,cuda_num=cuda_num)
net_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_model.items() if k in net_dict}
net_dict.update(pretrained_dict)
net.load_state_dict(net_dict)

for i,p in enumerate(net.parameters()):
    if i < 16:
        p.requires_grad = False #fix first 8 conv

if torch.cuda.is_available():
    net.cuda(cuda_num)

print('loaded...')


transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    ])

# prepare img data and labels, single process.
def imdata(im_paths):
    data_ = torch.zeros(len(im_paths),3,224,224)
    labels_ = torch.zeros(len(im_paths),12)
    landmarks_ = torch.zeros(len(im_paths),68,2)
    for i in range(len(im_paths)):
        im_path = im_paths[i]
        with Image.open(image_dir+'/'+im_path) as img:
            img = img.convert('RGB')
        img = transform(img)
        data_[i,:,:,:] = img
        labels_[i,:] = torch.tensor(labels[im_path.split('/')[-1]])
        landmarks_[i,:,:] = torch.reshape(torch.tensor(landmarks[im_path.split('/')[-1]]),(68,2))
    return data_,labels_,landmarks_

# learning rate and optimizer
learning_rate = 0.01
opt = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=learning_rate,momentum=0.9,nesterov=True)


for epoch in range(epochs):
    ### train
    # learning rate decay
    if epoch in [1,3,6,9]:
        learning_rate = learning_rate*0.1
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9,
                              nesterov=True)

    tmp_loss = 0                # tmp loss for every print_every iterations
    tmp_statistics_list = []    # tmp statistics list for every print_every iterations
    net.train()                 # train mode
    start = time.clock()        # timing
    for i in range(batches):
        i+=1
        selected_p = random.sample(train_person,person_batch) # random sample person_batch people/videos
        img_paths = []
        for each in selected_p:
            frames = random.sample(im_dic[each],seq_num)      # for each people/video, random sample seq_num images
            frames.sort()
            selected_frames = [each+'_'+str(kk)+'.jpg' for kk in frames]
            img_paths += selected_frames
        im_data,im_labels,im_landmarks = imdata(img_paths)
        if torch.cuda.is_available():
            im_data = im_data.cuda(cuda_num)
            im_labels = im_labels.cuda(cuda_num)
            im_landmarks = im_landmarks.cuda(cuda_num)
        pred = net(im_data,im_landmarks)
        loss = net.multi_label_ACE(pred,im_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        statistics_list = net.statistics(pred.data, im_labels.data, au_thresh)
        tmp_statistics_list = net.update_statistics_list(tmp_statistics_list,statistics_list)
        tmp_loss += loss

        if i%print_every == 0:
            mean_f1_score, f1_score_list = net.calc_f1_score(tmp_statistics_list)
            f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]
            elapsed = (time.clock()-start)
            logging.info('[TRAIN] epoch[%d/%d] batch[%d/%d] %.4f s/iter, loss:%.4f ave_loss:%.4f mean_f1_score:%.4f [%s]'
                 % (epoch+1, epochs, i,batches,elapsed/print_every,loss.data[0],tmp_loss/print_every, mean_f1_score, ' '.join(f1_score_list)))
            tmp_loss = 0
            tmp_statistics_list = []
            start = time.clock()

    torch.save(net.state_dict(), save_dir+'/params_epoch_%d.pkl' %epoch)  #save params

    ### test
    pred_test = torch.empty((0,12))
    y_test = torch.empty((0,12))
    if torch.cuda.is_available():
        pred_test = pred_test.cuda(cuda_num)
        y_test = y_test.cuda(cuda_num)
    loss_test = 0
    loss_count = 0
    im_names_ = []
    count = 0
    net.eval() #eval mode
    with torch.no_grad():
        for p in test_person:
            count += 1
            print('test: ', p, ' ', count, 'of', len(test_person))
            p_frames,len_p = im_dic[p],len(im_dic[p])
            for iii in range(int(len_p/seq_num)+1):
                if (len_p%seq_num != 0) and (iii==int(len_p/seq_num)):
                    mod_ = len_p % seq_num
                    selected_p = [p + '_' + str(kk) + '.jpg' for kk in p_frames[-1 * seq_num:]]
                    im_names_ += selected_p[-1 * mod_:]
                elif iii==int(len_p/seq_num):
                    continue
                else:
                    selected_p = [p+'_'+str(kk)+'.jpg' for kk in p_frames[iii*seq_num:(iii+1)*seq_num]]
                    im_names_ += selected_p
                im_data, im_labels, im_landmarks = imdata(selected_p)
                if torch.cuda.is_available():
                    im_data = im_data.cuda(cuda_num)
                    im_labels = im_labels.cuda(cuda_num)
                    im_landmarks = im_landmarks.cuda(cuda_num)
                pred_ts = net(im_data,im_landmarks)
                loss = net.multi_label_ACE(pred_ts,im_labels)
                loss_test += loss
                loss_count += 1
                if (len_p % seq_num != 0) and (iii == int(len_p / seq_num)):
                    pred_test = torch.cat((pred_test, pred_ts[-1 * mod_:, :]), dim=0)
                    y_test = torch.cat((y_test, im_labels[-1 * mod_:, :]), dim=0)
                else:
                    pred_test = torch.cat((pred_test,pred_ts),dim=0)
                    y_test = torch.cat((y_test,im_labels),dim=0)

    statistics_list_ = net.statistics(pred_test.data, y_test.data, au_thresh)
    mean_f1_score_, f1_score_list_ = net.calc_f1_score(statistics_list_)
    f1_score_list_ = ['%.4f' % f1_score for f1_score in f1_score_list_]
    logging.info('[TEST] epoch[%d/%d] loss: %.4f mean_f1_score:%.4f [%s]'
                 % (epoch + 1, epochs, loss_test/loss_count,mean_f1_score_, ' '.join(f1_score_list_)))
    save_path = save_dir+'/predictions_%d.txt' % epoch
    with open(save_path,'w') as f:
        for i in range(pred_test.size()[0]):
            f.write(im_names_[i]+' '+str(pred_test[i,:].cpu().numpy())+'\n')
    save_path = save_dir+'/labels_%d.txt' % epoch
    with open(save_path, 'w') as f:
        for i in range(y_test.size()[0]):
            f.write(im_names_[i] + ' ' + str(y_test[i, :].cpu().numpy()) + '\n')




