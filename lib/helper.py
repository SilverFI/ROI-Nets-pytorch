import os,json,random

# process all images names -> {'F001_T1':[2440,2441,...],'F002_T2':[],...,'F002_T3':[],...}
def get_im_dict():
    im_dic = {}
    with open('../data/all_images.txt') as f:
        all_im = f.readlines()
        for l in all_im:
            person = l.split('.')[0][0:7]
            num = int(l.split('.')[0][8:])
            if person not in im_dic:
                im_dic[person]=[]
                im_dic[person].append(num)
            else:
                im_dic[person].append(num)
        for k in im_dic.keys():
            im_dic[k].sort()
    return im_dic

# process labels.txt -> labels: {im_name:[labels]}, landmarks: {im_name:[landmark]}
def get_label_landmark():
    if not os.path.exists('../data/labels.json'):
        landmarks = {}
        labels = {}
        with open('../data/labels_224_landmarks.txt') as f:
            all_ = f.readlines()
            for l in all_:
                im_name, label, landmark = l.split('->')
                landmarks[im_name] = eval(landmark)
                labels[im_name] = eval(label)
        with open('../data/labels.json', 'w') as f:
            json.dump(labels, f)
        with open('../data/landmarks.json', 'w') as f:
            json.dump(landmarks, f)
    else:
        with open('../data/labels.json') as f:
            labels = json.load(f)
        with open('../data/landmarks.json') as f:
            landmarks = json.load(f)
    return labels,landmarks

# random split train/test data
def train_test_split():
    if not os.path.exists('../data/train.txt'):
        females = ['F%03d' % i for i in random.sample(range(1, 24), 23)]
        males = ['M%03d' % i for i in random.sample(range(1, 19), 18)]
        train_person = females[0:18] + males[0:15]
        train_person = [i + '_T' + str(j) for j in range(1, 9) for i in train_person]
        random.shuffle(train_person)
        test_person = females[18:] + males[15:]
        test_person = [i + '_T' + str(j) for j in range(1, 9) for i in test_person]
        random.shuffle(test_person)
        with open('../data/train.txt','w') as f:
            f.write(str(train_person))
        with open('../data/test.txt','w') as f:
            f.write(str(test_person))
    else:
        with open('../data/train.txt') as f:
            train_person = eval(f.readlines()[0])
        with open('../data/test.txt') as f:
            test_person = eval(f.readlines()[0])
    return train_person,test_person

