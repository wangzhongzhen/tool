
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pylab as plt
import random
from PIL import Image
import resnet_max_p_
import torch.optim as optim
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score,roc_curve,auc
import shutil


# In[2]:


# 验证集 模型
val_data = 'val_data'
model_path = 'maxp.t7'


# In[3]:


model = resnet_max_p_.my_net()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[4]:


checkpoint = torch.load(model_path)  #
model.load_state_dict(checkpoint['net'])
print(checkpoint['acc_c'])
print(checkpoint['acc_s'])
if torch.cuda.is_available():
    model.cuda()
model = model.eval()


# In[5]:


transform0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])


# In[6]:


def D(list):
    # 去掉mac中的Ds
    if '.DS_Store' in list:
            list.remove('.DS_Store')
    return list


# In[7]:


def default_loader(path,phase):
    path_ = os.path.join(path,'cpr')
    cpr_list = D(sorted(os.listdir(path_)))
    num = len(cpr_list)
    img = np.ones((4,60,60))
    
    if num==20:
        if phase == 'train':
        #start = random.randint(0,4)#torch
            start = int(torch.randint(0,4,(1,)))
        # print(start)
            for i in range(4):
                pic_cpr = cpr_list[start + i*5]
                pic_path = os.path.join(path_,pic_cpr)
    #             print(pic_path)
                img0 = cv2.imread(pic_path,0)

    #             img0 = Image.open(pic_path).convert('L')
                img[i] = img0
        else:
            for i in range(4):
                pic_cpr = cpr_list[0 + i*5]
                pic_path = os.path.join(path_,pic_cpr)
                #print(pic_path)
                img0 = cv2.imread(pic_path,0)
                img[i] = img0
                
    else:
        if phase == 'train':
            j = int(torch.randint(0,num,(1,)))
        else:
            j = 0
        #print(j)
        pic_path = os.path.join(path_,cpr_list[j])
        img_ = cv2.imread(pic_path,0)
        
        img = np.concatenate((img_[np.newaxis,:,:],img_[np.newaxis,:,:],img_[np.newaxis,:,:],img_[np.newaxis,:,:]),axis=0)
    return img


# In[8]:


class myImageFloder(Dataset):
    def __init__(self, workspace, default_loader, my_transforms = None, phase='train'):
        self.workspace = workspace
        self.loader = default_loader
        self.my_transform = my_transforms
        self.phase = phase
        img_id = []
        
        class_ = os.listdir(workspace)
        # 
        for label in class_:
            if label[0]=='.':
                continue
            path = os.path.join(workspace,label)   #train_data/2
            
            if path[0]=='.':
                continue
            for id_ in os.listdir(path):
                if id_[0]=='.' or id_.endswith('png'):
                    continue
                id_path = os.path.join(path,id_)
#                 print(id_path)
                
                img_id.append(id_path)
                
        self.img_id = img_id
#         for i,j in enumerate(range(len(self.img_id))):
#             print(i,self.img_id[j])
              
            
    def __getitem__(self,index):
        #print(index)
        img_id = self.img_id[index]
        img_label = img_id.split('/')[1]
#         print(img_id)
        img_ = self.loader(img_id,self.phase)
        img = np.zeros((4,60,60))      
        if self.my_transform is not None:
            
            for i in range(4):
                img[i] = self.my_transform(Image.fromarray(img_[i]).convert('L'))
                
        else:
            for i in range(4):
                img[i] = transform0(Image.fromarray(img_[i]).convert('L'))
#         plt.imshow(img[0])
#         plt.show()
        return img,int(img_label),img_id
    def __len__(self):
        return len(self.img_id)


# In[9]:


def loss1(output1,targets):
    # calc or not
    target = (((targets==2) + (targets==0))>0).long()
#     target = [0 if x==1 or x==3 else 1 for x in targets]
#     target = torch.tensor(target)
    
#     target = target.to(device)
#     print (output1.shape,target)
#     loss= criterion(output1,target)
    loss = 0
    return loss,target

def loss2(output1,targets):
    target = (((targets==1) + (targets==0))>0).long()

    # soft or not
#     target = [0 if x==2 or x==3 else 1 for x in targets]
#     target = torch.tensor(target)
#     target = target.to(device)
    return 0,target


# In[10]:


testset = myImageFloder(val_data,default_loader,phase = 'test')
testloader = torch.utils.data.DataLoader(testset,batch_size=300,shuffle=False,num_workers=2)


# In[ ]:


y_ture = np.array([1,1,0,0,1])
y_scores = np.array([0.5,0.6,0.55,0.4,0.7])


# In[ ]:


def Roc(y_ture,y_sorc):
    fpr,tpr,_ = roc_curve(y_ture,y_scores)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    plt.plot(fpr,tpr)
    plt.show()


# In[ ]:


Roc(y_ture,y_scores)


# In[ ]:


#  钙化曲线输出
#  测试badcase收集
# p 是指有病的概率
def run_test(epoch,p):

    print(p)
    save_path = os.path.join(output,str(10*p))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
#         assert False
    f = open(os.path.join(save_path,'calc.txt'),'w')
    px.append(p)
    model.eval()

    train_loss = 0
    correct_c = 0
    total_c = 0
    metrics_main0 = []
    metrics_main1 = []
    metrics_total_loss = []
    correct_s = 0
    total_s = 0
    metrics_main0_ = []
    metrics_main1_ = []
    for batch_idx, (inputs, targets,img_path) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)
        x = inputs.float().view((inputs.shape[0]*4,1)+inputs.shape[2:])

        p_c,p_s,idxc,idxs = model(x)

        pc = F.softmax(p_c)
        print('pc:',pc)
        ps = F.softmax(p_s)
        loss_c,target_c = loss1(p_c,targets)
        loss_s,target_s = loss2(p_s,targets)


        # cala  输出 0  是钙化    1 是非钙化
        predicted_c = (~(pc[:,0] > p)).long()

        fpr, tpr, thr = roc_curve(target_c.cpu().detach().numpy().ravel(),pc[:,0].cpu().detach().numpy().ravel(),pos_label=0)
        roc_auc = auc(fpr,tpr)
        print(roc_auc)

        plt.plot(fpr,tpr)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()
#         assert False
#         return fpr,tpr,thr
#             assert False
        print('predict:',predicted_c)
#         _,predicted_c = p_c.max(1)
        total_c += target_c.size(0)
        error_list = predicted_c.eq(target_c)
        print('target:',target_c)
        print('error_list:',error_list)
        print('pred',predicted_c.eq(target_c))
        for index,i in enumerate(error_list):
            if i == 0:     # 保存判断错的类别
                print('prob:',pc[index])
                print('cala error case:...........',img_path[index])
                print(img_path[index],save_path+'/'+img_path[index].split('/')[-1])
                shutil.copytree(img_path[index],save_path+'/'+img_path[index].split('/')[-1])
                f.write(img_path[index]+'    '+str(pc[index])+'\n')
        correct_c += predicted_c.eq(target_c).sum().item()

        
        metrics_main1.append(100.*correct_c/total_c)
    
           

        print('test','   Acc_c: %0.3f'%np.mean(metrics_main1),
            )
    return fpr,tpr,thr


# In[ ]:


p1 = 0.4863
px = []
py_c = []
output = 'badcase_c'
if not os.path.exists(output):
    os.makedirs(output)


# In[ ]:


fpr,tpr,thr = run_test(1,p1)


# In[ ]:


px.shape[0]


# In[ ]:


for i in range(px.shape[0]):
    print(thr[i])
    print('fpr',px[i],'tpr',py_c[i])
    plt.figure()
    plt.plot(px,py_c,'blue')
    plt.plot(px[:i],py_c[:i],'red')
    plt.show()


# In[11]:


# 软斑曲线图

def run_tests(epoch,p):
#     for p0 in range(-3,3):
#         p = p1+ p0/10
    print(p)
    save_paths = os.path.join(outputs,str(10*p))     # 每一个参数都放在单独文件夹内
    if not os.path.exists(save_paths):
        os.makedirs(save_paths)

    f = open(os.path.join(save_paths,'soft_error.txt'),'w')

    px.append(p)
    model.eval()

    train_loss = 0
    correct_c = 0
    total_c = 0
    metrics_main0 = []
    metrics_main1 = []
    metrics_total_loss = []
    correct_s = 0
    total_s = 0
    metrics_main0_ = []
    metrics_main1_ = []
    for batch_idx, (inputs, targets,img_path) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)
        x = inputs.float().view((inputs.shape[0]*4,1)+inputs.shape[2:])

        p_c,p_s,idxc,idxs = model(x)
#         print(p_c.shape,p_s.shape)
#         print('p_c:',p_c)
        pc = F.softmax(p_c)
#             print('pc:',pc)
        ps = F.softmax(p_s)
        print('ps:',ps)
        loss_c,target_c = loss1(p_c,targets)
        loss_s,target_s = loss2(p_s,targets)


        # cala  输出 0  是钙化    1 是非钙化
        predicted_c = (~(pc[:,0] > p)).long()
#             print('predict:',predicted_c)
#         _,predicted_c = p_c.max(1)
        total_c += target_c.size(0)
        error_list = predicted_c.eq(target_c)
#             print('target:',target_c)
#         print('pred',predicted_c.eq(target_c))
        for index,i in enumerate(error_list):
            if i == 0:
                None
#                     print('cala error case:...........',img_path[index])

        correct_c += predicted_c.eq(target_c).sum().item()

#         metrics_main0.append(loss_c.item())
        metrics_main1.append(100.*correct_c/total_c)


        # soft

        predicted_s = (~(ps[:,0] > p)).long()
        error_list_s = predicted_s.eq(target_s)
        print('target:',target_s)
        print('pred',predicted_s)
        # plot roc
        fpr, tpr, thr = roc_curve(target_s.cpu().detach().numpy().ravel(),ps[:,0].cpu().detach().numpy().ravel(),pos_label=0)
        roc_auc = auc(fpr,tpr)
        print(roc_auc)
        print(_)
        plt.plot(fpr,tpr)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()
#         assert False
#     return fpr,tpr,thr
#             print('predict:',predicted_c)
#             _, predicted_s = p_s.max(1)
        total_s += target_s.size(0)
        for index,i in enumerate(error_list_s):
            if i == 0:
                print('soft error case:...........',img_path[index])
                print('prob:',ps[index])
                shutil.copytree(img_path[index],save_paths+'/'+img_path[index].split('/')[-1])
                f.write(img_path[index]+'    '+str(ps[index])+'\n')
            else:
                print('soft ok',img_path[index])
                print('prob:',ps[index])
#                     print(img_path[index],save_paths+'/'+img_path[index].split('/')[-1])
#                     shutil.copytree(img_path[index],save_paths+'/'+img_path[index].split('/')[-1])
#                     f.write(img_path[index]+'    '+str(ps[index])+'\n')
                if p == 0.3:
                    None
#                         shutil.copytree(img_path[index],outputs+'/'+img_path[index].split('/')[-1])
        correct_s += predicted_s.eq(target_s).sum().item()

#         metrics_main0_.append(loss_s.item())
        metrics_main1_.append(100.*correct_s/total_s)
#         metrics_total_loss.append(loss.item())

    print('test','  loss_c: %.3f'% np.mean(metrics_main0),'   Acc_c: %0.3f'%np.mean(metrics_main1),
         '  loss_s: %.3f'% np.mean(metrics_main0_),'   Acc_s: %0.3f'%np.mean(metrics_main1_))
    return fpr,tpr,thr


# In[12]:


p1 = 0.492
px = []
py_c = []
py_s = []
outputs = 'badcase_s'
if not os.path.exists(outputs):
    os.makedirs(outputs)


# In[13]:


# 软斑测试
fpr,tpr,thr = run_tests(1,p1)


# In[14]:


fpr.shape


# In[15]:


for i in range(35):
    print(thr[i])
    print('fpr',fpr[i],'tpr',tpr[i])
    plt.figure()
    plt.plot(fpr,tpr,'blue')
    plt.plot(fpr[:i],tpr[:i],'red')
    plt.show()


# In[ ]:



plt.plot(px,py_s)
plt.show()
print(py_s)


# In[ ]:


def test(epoch):
    global clr,best_loss
    #print('test...',epoch)
    model.eval()
    
    train_loss = 0
    correct_c = 0
    total_c = 0
    metrics_main0 = []
    metrics_main1 = []
    metrics_total_loss = []
    correct_s = 0
    total_s = 0
    metrics_main0_ = []
    metrics_main1_ = []
    for batch_idx, (inputs, targets,img_path) in enumerate(testloader):
#         print(img_path)
#         print(inputs.shape, targets.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        x = inputs.float().view((inputs.shape[0]*4,1)+inputs.shape[2:])

        p_c,p_s,idxc,idxs = model(x)
#         print(p_c.shape,p_s.shape)
#         print('p_c:',p_c)
        pc = F.softmax(p_c)
        print('pc:',pc)
        ps = F.softmax(p_s)
        loss_c,target_c = loss1(p_c,targets)
        loss_s,target_s = loss2(p_s,targets)

        
        # cala  输出 0  是钙化    1 是非钙化
        predicted_c = (~(pc[:,0] > p)).long()
        print('predict:',predicted_c)
#         _,predicted_c = p_c.max(1)
        total_c += target_c.size(0)
        error_list = predicted_c.eq(target_c)
        print('target:',target_c)
#         print('pred',predicted_c.eq(target_c))
        for index,i in enumerate(error_list):
            if i == 0:
                print('cala error case:...........',img_path[index])
                
        correct_c += predicted_c.eq(target_c).sum().item()
        
#         metrics_main0.append(loss_c.item())
        metrics_main1.append(100.*correct_c/total_c)
        

        # soft
        
        _, predicted_s = p_s.max(1)
        total_s += target_s.size(0)
        correct_s += predicted_s.eq(target_s).sum().item()
        
#         metrics_main0_.append(loss_s.item())
        metrics_main1_.append(100.*correct_s/total_s)
#         metrics_total_loss.append(loss.item())
        
    print('test','  loss_c: %.3f'% np.mean(metrics_main0),'   Acc_c: %0.3f'%np.mean(metrics_main1),
         '  loss_s: %.3f'% np.mean(metrics_main0_),'   Acc_s: %0.3f'%np.mean(metrics_main1_))





# In[ ]:


def 


# In[ ]:


test(1)


# In[ ]:


p_c = np.array([[1,  3],
        [ 2,  0],
        [3,  4]])


# In[ ]:


a = p_c!=2


# In[ ]:


a


# In[ ]:


b = p_c!=0


# In[ ]:


b


# In[ ]:


c = a * b


# In[ ]:


c


# In[ ]:


p_c[c] = 8


# In[ ]:


p_c


# In[ ]:


p_c = torch.from_numpy(p_c)


# In[ ]:


pc = F.softmax(p_c,1)


# In[ ]:


pc


# In[ ]:


pc.max(1)


# In[ ]:


a = ~(pc[:,0] > 0.4)


# In[ ]:


a


# In[ ]:


a == 0


# In[ ]:


a 


# In[ ]:


b = ['a','b','c']


# In[ ]:


for index,i in enumerate(a):
    if i == 0:
        print(b[index])


# In[ ]:


import shutil


# In[ ]:


shutil.rmtree('badcase_s')

