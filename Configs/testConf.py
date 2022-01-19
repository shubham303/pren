from easydict import EasyDict as edict

configs = edict()

configs['cuda'] = True

#configs['image_dir'] = '/home/yrj/Dataset/SceneText/English/Test_sets/IIIT5K/'
#configs['val_list'] = "/home/yrj/Dataset/SceneText/English/Test_sets/IIIT5K/test.txt"

#onfigs['image_dir'] = '/home/yrj/Dataset/SceneText/English/Test_sets/SVT/'
#onfigs['val_list'] = "/home/yrj/Dataset/SceneText/English/Test_sets/SVT/test.txt"

#configs['image_dir'] = '/home/yrj/Dataset/SceneText/English/Test_sets/ICDAR2003/crop/'
#configs['val_list'] = "/home/yrj/Dataset/SceneText/English/Test_sets/ICDAR2003/crop/gt_867.txt"

#onfigs['image_dir'] = '/home/yrj/Dataset/SceneText/English/Test_sets/ICDAR2013/'
#onfigs['val_list'] = '/home/yrj/Dataset/SceneText/English/Test_sets/ICDAR2013/label_857.txt'

#configs['image_dir'] = '/home/yrj/Dataset/SceneText/English/Test_sets/ICDAR2015/'
#configs['val_list'] = "/home/yrj/Dataset/SceneText/English/Test_sets/ICDAR2015/gt_1811.txt"

#configs['image_dir'] = '/home/yrj/Dataset/SceneText/English/Test_sets/SVT-Perspective/'
#configs['val_list'] = "/home/yrj/Dataset/SceneText/English/Test_sets/SVT-Perspective/gt.txt"

configs['image_dir'] = [
                        "/usr/datasets/synthetic_text_dataset/lmdb_dataset_Hindi/hindi/evaluation/IIIT"]

configs['val_list'] = '/home/yrj/Dataset/SceneText/English/Test_sets/CUTE80/crop/gt.txt'

configs['model_path'] = './models/20220119-13:45:24/m_epoch1_22000.pth'

configs['imgH'] = 64
configs['imgW'] = 256
configs['alphabet'] = "data/alphabet_hi.txt"

configs['vert_test'] = False  # if the image has its height > width, rotate it and choose the prediction with the
# highest confidence
configs['batchsize'] = 512  # if vert_test, batchsize should be 1
configs['display'] = True  # display the prediction of each image
configs['workers'] = 0
