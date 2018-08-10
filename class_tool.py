#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:06:27 2018

@author: apple
"""

import matplotlib.pylab as plt
import os
import cv2
import os


def get_input():
    print('input label:')
    l = input()
    
    return l


workspace = 'classlabel'
out_csv = 'out2.csv'

def show(path):
    fig = plt.figure(figsize=(10,10))
    column = 5
    rows = 4
    for i,id in enumerate(os.listdir(path),1):
        if id[0]=='.':
            continue
        show_path = os.path.join(path,id)
        mat = cv2.imread(show_path,0)
        fig.add_subplot(rows,column,i)
#         plt.imshow(mat)
        plt.imshow(mat,cmap=plt.get_cmap('gray'), interpolation='nearest')
    plt.show()
        

def main(workspace):
    f = open(out_csv,'w')
    for i,label0 in enumerate(sorted(os.listdir(workspace)),0):
        if label0[0] == '.':
            continue
        path = os.path.join(workspace,label0)
        for id in os.listdir(path):
            if id[0] == '.':
                continue
            show_path = os.path.join(os.path.join(path,id),'cpr')
            
            try:
                show(show_path)
                print(show_path)
                man_label = get_input()
                print (os.path.join(path,id) +'..'+str(man_label))
                f.write(os.path.join(path,id) +'..'+str(man_label)+'\n')
            except:
                None
    f.close()
main(workspace)


