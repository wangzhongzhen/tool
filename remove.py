
# coding: utf-8

# In[8]:


import os
import sys


# In[11]:


# os.system('rm -rf '+'test/P00010320T20180419R165054/center_line_tmp')
workspace = sys.argv[1]


# In[15]:


# workspace = 'test'
for id in os.listdir(workspace):
    if id[0]=='.':
        continue
    case_path = os.path.join(workspace,id)
    for ids in os.listdir(case_path):
        if ids=='narrow_list' or ids=='cpr' or ids=='shortaxis':
            continue
        else:
            path =  os.path.join(case_path,ids)
            print path
            os.system('rm -rf ' + path)

