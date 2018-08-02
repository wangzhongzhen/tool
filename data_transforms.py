
# coding: utf-8

# In[1]:


import numpy as np
import random
import os
import matplotlib.pylab as plt

import scipy


# In[29]:


workspace = '0723_test'
output = '0723_test_arg'


# In[ ]:


def shift_arg(data):
    re_data = []
    size,_,__ = data.shape
    
    mul_data = np.pad(data,((size,size),(0,0),(0,0)),'reflect')
    np.random.shuffle(mul_data)
    
#     re_data.append(data)
    re_data.append(mul_data[:size])
    re_data.append(mul_data[size:size*2])
    re_data.append(mul_data[size*2:])
    return re_data


# In[47]:


def random_rotation_3d(batch,max_angle=10):
    
    batch_rot = np.zeros(batch.shape)
    
    # rotate along z-axis
    angle = random.uniform(-max_angle,max_angle)
    img2 = scipy.ndimage.interpolation.rotate(batch,angle,mode='nearest',axes=(0,1),reshape=False)
    # rotate along y-axis
    angle = random.uniform(-max_angle, max_angle)
    img3 = scipy.ndimage.interpolation.rotate(img2, angle, mode='nearest', axes=(0, 2), reshape=False)

    # rotate along x-axis
    angle = random.uniform(-max_angle, max_angle)
    batch_rot = scipy.ndimage.interpolation.rotate(img3, angle, mode='nearest', axes=(1, 2), reshape=False)
    return batch_rot
# input = np.random.rand(30,30,30)

# bat = random_rotation_3d(input)
# print(bat.shape)


# In[50]:


def main(workspace):
    show = 0
    arg_data = 0
    rotate_data = 1
    for label in os.listdir(workspace):
        path = os.path.join(workspace,label)
        for id in os.listdir(path):
            if id[0]=='.':
                continue
            data = np.load(os.path.join(path,id))#[np.newaxis]
            if arg_data==1:
                # shift arg
                arg_data = shift_arg(data)

                for i in range(len(arg_data)):
                    outpath = os.path.join(path,id.split('.')[0]+'.%s.npy'%i)
                    np.save(outpath,arg_data[i])
            if rotate_data==1:
                # rotate arg
                print('data for rotate shape:',data.shape)
                arg_data_rotate = random_rotation_3d(data)

                outpath = os.path.join(path,id.split('.')[0]+'.rot.npy')

                np.save(outpath,arg_data_rotate)
            
            
            if show == 1:
                show_data = arg_data_rotate[0]
                plt.imshow(show_data)
                plt.show()
               
main(workspace)
        

