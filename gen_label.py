#-*- coding:utf-8 -*-
import os
import sys
import cv2


os.system("find train/pos  -name '*.jpg' > pos.txt" )


fp = open('pos_label.txt', 'w')

for in_line in open('pos.txt'):
    in_line = in_line[:-1]
    fp.write(in_line + ' 1\n')

fp.close()

os.system("find train/neg  -name '*.jpg' > neg.txt" )


fp = open('neg_label.txt', 'w')

for in_line in open('neg.txt'):
    in_line = in_line[:-1]
    fp.write(in_line + ' 0\n')

fp.close()

os.system("cat pos_label.txt neg_label.txt > label_tmp.txt" )
os.system("shuf label_tmp.txt > label.txt" )

p = os.popen('wc -l label.txt')
sample_num = p.read()
sample_num = sample_num.split(' ')[0]
sample_num = int(sample_num)
p.close()

valid_num =  sample_num * 20 / 100

os.system('head -n' + str(valid_num) + ' label.txt  > val_label.txt' )
os.system('tail -n' + str(sample_num - valid_num) + ' label.txt > train_label.txt' )

os.system('rm -rf pos.txt neg.txt neg_label.txt pos_label.txt label.txt label_tmp.txt train.lmdb val.lmdb')

os.system('/opt/caffe-shukun/build/tools/convert_imageset ./ label.txt --shuffle=true --resize_width=224 --resize_height=224 train.lmdb')
os.system('/opt/caffe-shukun/build/tools/convert_imageset ./ label.txt --shuffle=true --resize_width=224 --resize_height=224 val.lmdb')

os.system('/opt/caffe-shukun/build/tools/compute_image_mean val.lmdb val.mean')
os.system('/opt/caffe-shukun/build/tools/compute_image_mean train.lmdb train.mean')




