
# coding: utf-8

# In[11]:


import zipfile
import shutil
import os


# In[27]:


shutil.rmtree('cpr_test_zip/')


# In[31]:


workspace = "dcm_cpr_all/"
dest = "dcm_zip"
if not os.path.exists(dest):
    os.makedirs(dest)


# In[33]:


for i,id in enumerate(os.listdir(workspace),0):
    ID = os.path.join(workspace,id) 
    if id[0] == '.'or id[-1]=='p':
        continue
    print(i,id)   
    if i%400 == 0:
       print(os.path.join(dest,"{}.zip".format(i)))
       f = zipfile.ZipFile(os.path.join(dest,"{}.zip".format(i)),'w',zipfile.ZIP_DEFLATED)
    for pic in os.listdir(ID):
       pic_path = os.path.join(ID,pic)
       print(pic_path)
       f.write(pic_path)
f.close


# In[26]:


# cmd
os.getcwd()


# In[25]:


os.chdir("..")


# In[10]:


for i,id in enumerate(os.listdir("./"),0):
    if id[0] == '.':
        continue
    print(id)

