
# coding: utf-8

# In[2]:


import os
from PIL import Image
import shutil


# In[49]:


# shutil.rmtree(destfile)
# shutil.move('')
# 对原始的ID数据提取有病的血管，并保存


# In[51]:


workspace = 'dcm_workspace'
destfile = 'dcm_cpr_all'
if not os.path.exists(destfile):
    os.mkdir(destfile)


# In[52]:


B_TABLE = {
        'pRCA' : 'RCA',
        'mRCA' : 'RCA',
        'dRCA' : 'RCA',
        'R-PDA' : 'R-PDA',
        'R-PLB' : 'R-PLB',
        'RI' : 'RI',
        'LM' : 'LAD',
        'pLAD' : 'LAD',
        'mLAD' : 'LAD',
        'dLAD' : 'LAD',
        'D1' : 'D1',
        'D2' : 'D2',
        'pCx' : 'LCX',
        'LCx' : 'LCX',
        'OM1' : 'OM1',
        'OM2' : 'OM2',
                      }


# In[53]:


for case in os.listdir(workspace):
    if case[0]=='.':
        continue
    case_path = os.path.join(workspace,case)
    narrow_path = os.path.join(case_path,'narrow_list','narrow_result_classified.csv')
    if not os.path.exists(narrow_path):
#         print(case,'no exitst narrow_result')
        continue
    work_path = os.path.join(case_path,'cpr_dcm')
    if not os.path.exists(work_path):
        print(case,'no exitst cpr')
        continue
    vname = []
    with open(narrow_path) as f:
        for line in f.readlines():
            line = line.strip()
            data = line.split(',')
            if len(data[0]) == 0:
                break

            vname.append(B_TABLE[data[1]])
#     print("before",vname)
    vname = list(set(vname))
#     print("after",vname)
    for vname_ in  vname:      
        section = vname_
        orig_section = os.path.join(work_path,section,'noline') + '/'
#         orig_txt = os.path.join(work_path,section,)
#             print(orig_section)
        new_name = case+'_'+section+'_'
        dest_new_name = os.path.join(destfile,new_name)

        print(orig_section,'  ',dest_new_name)
        shutil.copytree(orig_section,dest_new_name)
        
    print('____________________')


# In[33]:


import shutil


# In[28]:



# shutil.rmtree(destfile)

