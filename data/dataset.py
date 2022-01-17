import os
import sys
import random

import cv2
import lmdb
import numpy as np
import six
from PIL import Image

from data.data_utils import Augmenter

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class lmdbDataset(Dataset):

    def __init__(self, roots=None, ratio=None, img_height = 32, img_width = 128,
        transform=None, inTrain=True):
        self.envs = []
        self.nSamples = 0
        self.lengths = []
        self.ratio = []
        self.inTrain = inTrain
        for i in range(0,len(roots)):
            env = lmdb.open(
                roots[i],
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
            if not env:
                print('cannot creat lmdb from %s' % (roots))
                sys.exit(0)
            with env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                self.nSamples += nSamples
            self.lengths.append(nSamples)
            self.envs.append(env)

        if ratio != None:
            assert len(roots) == len(ratio) ,'length of ratio must equal to length of roots!'
            for i in range(0,len(roots)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0,len(roots)):
                self.ratio.append(self.lengths[i] / float(self.nSamples))

        self.transform = transform
        self.maxlen = max(self.lengths)
        #self.img_height = img_height
        #elf.img_width = img_width
        self.target_ratio = img_width / float(img_height)
        self.p_aug = 0
        if inTrain:
            self.aug = Augmenter(p=self.p_aug)

        self.m = "ऀ  ँ ं ः  ॕ "
        self.V = "ऄ ई ऊ ऍ  ऎ ऐ ऑ ऒ ओ औ"
        self.CH = "अ आ उ ए इ ऌ क  ख  ग ऋ  घ  ङ  च  छ  ज  झ  ञ  ट  ठ  ड  ढ  ण  त  थ  द  ध  न  ऩ  प  फ  ब  भ  म  य  र  " \
                  "ऱ  ल  ळ  ऴ  व  " \
             "श  ष  " \
             "स  ह ॐ क़  ख़  ग़  ज़  ड़  ढ़  फ़  य़  ॠ  ॡ"
        self.v = "ा  ि  ी  ु  ू  ृ  ॄ  ॉ  ॊ  ो  ौ  ॎ  ॏ ॑  ॒  ॓ ़ ॔  ॅ े ै ॆ ्  ॖ   ॗ ॢ  ॣ"
        self.symbols = "।  ॥  ०  १  २  ३  ४  ५  ६  ७  ८  ९"

    def __fromwhich__(self ):
        rd = random.random()
        total = 0
        for i in range(0,len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i

    def keepratio_resize(self, img):
        cur_ratio = img.size[0] / float(img.size[1])

        mask_height = self.img_height
        mask_width = self.img_width
        img = np.array(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if cur_ratio > self.target_ratio:
            cur_target_height = self.img_height
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))
        start_x = int((mask_height - img.shape[0])/2)
        start_y = int((mask_width - img.shape[1])/2)
        mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
        mask[start_x : start_x + img.shape[0], start_y : start_y + img.shape[1]] = img
        img = mask
        return img

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.inTrain:
            index = random.randint(0,self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode()).decode('utf-8')
            if (len(label) >= 25 and self.inTrain) or (not self.is_valid_label(label)):
                # print('Too long text: {}, use next one.'.format(imgpath))
                return self[index + 1]
            """try:
                img = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]"""
            h, w,_= img.shape
            if min(h, w) <= 5:
                # print('Too small image {}, use next one.'.format(imgpath))
                return self[index + 1]
            
            if self.inTrain:
                img = self.aug.apply(img, len(label))
                
            #img = img[:,:,np.newaxis]
            if self.transform:
                img = self.transform(img)
            sample = {'image': img, 'label': label}
            return sample

    def is_valid_label(self, label):
        # ref https://www.unicode.org/L2/L2016/16161-indic-text-seg.pdf
        
        state = 0
        valid = True
        
        for ch in list(label):
        
            if not (ch in self.v or ch in self.m or ch in self.V or ch in self.CH or ch in self.symbols):
                return False
        
            if ch in self.symbols:
                state = 0
                continue
        
            if ch in self.CH:
                state = 2
                continue
        
            if ch in self.V:
                state = 1
                continue
        
            if state == 0:
                if ch in self.v or ch in self.m:
                    valid = False
                    break
        
            if state == 1:
                if ch in self.v:
                    valid = False
                    break
            
                if ch in self.m:
                    state = 0
                    continue
        
            if state == 2:
                if ch in self.v:
                    state = 3
                    continue
            
                if ch in self.m:
                    state = 0
                    continue
        
            if state == 3:
                if ch in self.m:
                    state = 0
                    continue
            
                if ch in self.v:
                    valid = False
                    break
        return valid


class listDataset(Dataset):
    def __init__(self, imgdir=None, list_file=None, transform=None, inTrain=False, p_aug=0, vert_test=False):
        '''
        :param imgdir: path to root directory
        :param list_file: path to ground truth file
        :param transform: torchvison transforms object
        :param inTrain: True for training stage and False otherwise
        :param p_aug: probability of data augmentation
        '''

        self.list_file = list_file
        with open(list_file) as fp:
            self.lines = fp.readlines()
            self.nSamples = len(self.lines)

        self.transform = transform
        self.imgdir = imgdir
        self.inTrain = inTrain
        self.p_aug = p_aug
        self.vert_test = vert_test

        if inTrain:
            self.aug = Augmenter(p=self.p_aug)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # -- get image
        line_splits = self.lines[index].split()
        imgpath = os.path.join(self.imgdir, line_splits[0])

        img = cv2.imread(imgpath)

        # ignore invalid images
        if img is None:
            #print('Invalid image {}, use next one.'.format(imgpath))
            return self[index + 1]

        # ignore too small images
        h, w, _ = img.shape
        if min(h, w) <= 5:
            # print('Too small image {}, use next one.'.format(imgpath))
            return self[index + 1]

        # -- get text label
        label = ' '.join(line_splits[1:])
        label = label.lower()

        # ignore too long texts in training stage
        if len(label) >= 25 and self.inTrain:
            # print('Too long text: {}, use next one.'.format(imgpath))
            return self[index + 1]

        # -- data preprocess
        if self.inTrain:
            img = self.aug.apply(img, len(label))

        x = self.transform(img)
        x.sub_(0.5).div_(0.5)  # normalize to [-1, 1)

        # for vertical test samples, return rotated versions
        x_clock, x_counter = 0, 0
        is_vert = False
        if self.vert_test and not self.inTrain and h > w:
            is_vert = True
            img_clock = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_counter = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_clock = self.transform(img_clock)
            x_counter = self.transform(img_counter)
            x_clock.sub_(0.5).div_(0.5)
            x_counter.sub_(0.5).div_(0.5)

        return (x, label, x_clock, x_counter, is_vert, imgpath)


def TrainLoader(configs):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.Resize((configs.imgH, configs.imgW)),
        transforms.ToTensor()
    ])
    
    dataset = lmdbDataset(roots=configs.image_dir,
                          transform=transform,
                          inTrain=True)
    
    """dataset = listDataset(imgdir=configs.image_dir,
                          list_file=configs.train_list,
                          transform=transform,
                          inTrain=True,
                          p_aug=configs.aug_prob)"""

    return DataLoader(dataset,
                      batch_size=configs.batchsize,
                      shuffle=True,
                      num_workers=configs.workers)


def TestLoader(configs):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((configs.imgH, configs.imgW)),
        transforms.ToTensor()
    ])
    
    dataset= lmdbDataset(roots=configs.image_dir,
                         transform=transform,
                         inTrain=False)

    """  dataset = listDataset(imgdir=configs.image_dir,
                          list_file=configs.val_list,
                          transform=transform,
                          inTrain=False,
                          vert_test=configs.vert_test)"""

    return DataLoader(dataset,
                      batch_size=configs.batchsize,
                      shuffle=False,
                      num_workers=configs.workers)


if __name__== '__main__':

    from Configs.trainConf import configs
    import matplotlib.pyplot as plt

    train_loader = TrainLoader(configs)
    l = iter(train_loader)
    im, la, *_ = next(l)
    for i in range(100):
        plt.imshow(im[i].permute(1,2,0) * 0.5 + 0.5)
        plt.show()

    # import matplotlib.pyplot as plt
    # from Configs.testConf import configs
    # valloader = TestLoader(configs)
    # l = iter(valloader)
    # im, la, *_ = next(l)
    # plt.imshow(im[0].permute(1, 2, 0) * 0.5 + 0.5)

