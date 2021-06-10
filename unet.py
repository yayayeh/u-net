import json
import cv2
import glob
import os

from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# detect image size
train_dir = 'data/train/image/'
train_list = os.listdir(train_dir)
train_name_list = []
train_scale = []

for train_path in train_list:
	if train_path.split('.')[-1] == 'jpg':
	    train_name = train_path.split('.')[0][4:]
	    # print(train_name)
	    train_name_list.append(train_name)
	    train_name_list.sort(key = int)

for i in range(len(train_name_list)):
	train_name_list[i] = 'img_'+train_name_list[i]+'.jpg'

for train_path in train_list:
	if train_path.split('.')[-1] == 'jpg':
	    data = cv2.imread(train_dir + train_path)
	    in_size = []
	    in_size.append(data.shape[1]/256)
	    in_size.append(data.shape[0]/256)
	    train_scale.append(in_size)
print("train_scale:",train_scale)


test_dir = 'data/test/'
test_list = os.listdir(test_dir)
test_name_list = []
test_scale = []

for test_path in test_list:
	if test_path.split('.')[-1] == 'jpg':
	    test_name = test_path.split('.')[0][4:]
	    test_name_list.append(test_name)
	    test_name_list.sort(key = int)

for i in range(len(test_name_list)):
	test_name_list[i] = 'img_'+test_name_list[i]+'.jpg'

for test_path in test_list:
	if test_path.split('.')[-1] == 'jpg':
	    data = cv2.imread(test_dir + test_path)
	    # print("data.shape",data.shape)
	    in_size = []
	    in_size.append(data.shape[1]/256)
	    in_size.append(data.shape[0]/256)
	    # print('in_size:',in_size)
	    test_scale.append(in_size)
print("test_scale:",test_scale)
# detect image size end

# change image color
def togrey(img,outdir):
    src = cv2.imread(img) 
    try:
        dst = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        #cv2_imshow(dst)
        cv2.imwrite(os.path.join(outdir,os.path.basename(img)), dst)
    except Exception as e:
        print(e)

for file in glob.glob('data/train/image/*.jpg'):
	togrey(file,'data/train/image/')

for file in glob.glob('data/test/*.jpg'): 
	togrey(file,'data/test/')
# change image color end

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=200,epochs=80,callbacks=[model_checkpoint])

testGene = testGenerator("data/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/test/predict",results)

# resize predicted results
for i in range(len(test_name_list)):
  img = cv2.imread("data/test/predict/" + test_name_list[i].split('.')[0][4:] + "_predict.png")
  print('original image shape:', img.shape)

  height, width, channel = img.shape
  resized_img = cv2.resize(img, (int(width * test_scale[i][1]), int(height * test_scale[i][0])), interpolation=cv2.INTER_CUBIC)
  print('resized to image shape:', resized_img.shape)
  os.chdir('data/test/resize')
  save_name = test_name_list[i].split('.')[0] + "_resize.png"
  cv2.imwrite(save_name, resized_img) 