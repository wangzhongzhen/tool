
# coding: utf-8

# In[1]:


# shuffle data


# In[2]:


import random
import os


# In[4]:


workspace = './data/'


# In[34]:


def shuffle_fun(workspace):
    os.system("find data -name '*_label.png' > all.txt")
    os.system('shuf all.txt > shu_all.txt')
    
    f = open('label.txt','w')
    for lines in open('shu_all.txt'):
        line = lines.replace('_label.png','').replace('data/','')
        f.write(line+'\n')
    f.close
    
    p = os.popen('wc -l shu_all.txt')
    sample_num = p.read()
    
    sample_num = sample_num.split(' ')[0]
    sample_num = int(sample_num)
    p.close()
    
    os.system('rm all.txt shu_all.txt')
    valid_num =  int(sample_num * 20 / 100)
    print(valid_num)
    os.system('head -n' + str(valid_num) + ' label.txt  > val_label.txt' )
    os.system('tail -n' + str(sample_num - valid_num) + ' label.txt > train_label.txt' )



# In[35]:


shuffle_fun(workspace=workspace)

