import torch
from step_fun import Embbed
from torch.nn.parameter import Parameter
import numpy as np
from PIL import Image
from model import HPF, HILLSTC, HILLCOST, HPF_SRM
from PIL import Image
from cfg import *
from hpfloss import HPFLOSS
os.system('mkdir data_at')
start_num = 0
class ImageNetTest(Dataset):
    def __init__(self, path,names):
        self.path = path
        self.names = names
    def __getitem__(self, item):
        item = item + start_num
        name = self.names[item].replace('\n','')
        img = cft_transform(Image.open(self.path + name))
        label = name.split('_')[1]
        label = label.replace('.JPEG',' ')
        label = torch.tensor(int(label)).to(dtype=torch.long)
        return img,label
    def __len__(self):
        return 1000

file=open('../names.txt', 'r')
list_read = file.readlines()

normal_data = ImageNetTest('./ILSVRC2012_img_val_labeled/',list_read)
normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False,num_workers=num_workers)


HILLCOST = HILLCOST().to(device)
HPF = HPF_SRM().to(device)
HPF.eval()

model = nn.Sequential(
    norm_layer,
    models.inception_v3(pretrained=True)
).to(device)


model = model.eval()

def f(x,model,labels):
    outputs = model(x)
    one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())

    return j-i

def saveimg(x,name):
    global count_adv
    l = x.shape[0]
    for i in range(l):
        img = x[i,:,:,:].transpose(1,2,0)
        img = Image.fromarray(img)
        if name == 'at':
            count_adv += 1 
            img.save('./data_at/'+str(count_adv)+'.bmp')
        elif name == 'ut':
            count_adv += 1
            img.save('./data_ut/'+str(count_adv)+'.bmp')


def zn_attack(images,labels,order,c=20,maxstep=5000):
    if order == 3:
        target = images.view(-1,1,img_size,img_size)*255
        HPFtarget = HPF(target).reshape(-1,1,img_size,img_size)
        cost = HILLCOST(HPFtarget)
        cost[cost > 1] = 1
        cost = cost.detach()
        test_image = images.view(-1,1,img_size,img_size)*255
        map = Parameter(0.5 * torch.zeros(batch_size, 18, img_size, img_size).to(device))
        optimizer = torch.optim.Adam([map], lr=1e-2)
        for ii in range(maxstep):
            map.data.clamp_(0, 0.5)
            P = map[:, 0:3, :, :]
            M = map[:, 3:6, :, :]
            P_2 = map[:, 6:9, :, :]
            M_2 = map[:, 9:12, :, :]
            P_3 = map[:, 12:15, :, :]
            M_3 = map[:, 15:18, :, :]
            rand = torch.rand_like(images).reshape(-1, 1)
            rand2 = torch.rand_like(images).reshape(-1, 1)
            rand3 = torch.rand_like(images).reshape(-1, 1)
            temp1 = torch.cat((rand, P.reshape(-1, 1), M.reshape(-1, 1)), 1)
            temp2 = torch.cat((rand2, P_2.reshape(-1, 1), M_2.reshape(-1, 1)), 1)
            temp3 = torch.cat((rand3, P_3.reshape(-1, 1), M_3.reshape(-1, 1)), 1)
            adv_img = Embbed(temp1).view(-1, 1, img_size, img_size)*1 + Embbed(temp2).view(-1, 1, img_size, img_size)*2 + Embbed(temp3).view(-1, 1, img_size, img_size)*3 + test_image
            
            adv_img = adv_img.view(-1,3,img_size,img_size)

            t_a = adv_img.view(-1,1,img_size,img_size)
            HPFa = HPF(t_a).reshape(-1,1,img_size,img_size)
            losshpf = criterionl1(cost * HPFa, cost * HPFtarget.detach())


            out = model(adv_img/255)
            _, pre = torch.max(out.data, 1)
            correct_train = (pre == labels).sum()
            # if pre.item() == labels.item():
            #     break
            CELOSS = CEL(out,labels)
            loss = CELOSS + c*losshpf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp = torch.sum(f(adv_img/255,model,labels)).item()
            # if ii % 100 == 0:
            #     print('step:{} order:{} loss:{} tmp:{} hpf:{}'.format(ii,order,CELOSS.item(),tmp,losshpf.item()))         
            if torch.sum(f(adv_img/255,model,labels)).item() > 0.001*batch_size:
                tmp = adv_img.clone().detach()
                tmp[tmp>255]=255
                tmp[tmp<0]=0
                tmp = (torch.round(tmp)).type(torch.uint8)
                outputs_white = model(tmp.type(torch.float)/255)
                _, pre_white = torch.max(outputs_white.data, 1)
                if pre_white == labels:
                    return adv_img
        return adv_img
    elif order == 1:
        target = images.view(-1,1,img_size,img_size)*255
        HPFtarget = HPF(target).reshape(-1,1,img_size,img_size)
        cost = HILLCOST(HPFtarget)
        cost[cost > 1] = 1
        cost = cost.detach()
        test_image = images.view(-1,1,img_size,img_size)*255
        map = Parameter(0.5 * torch.zeros(batch_size, 6, img_size, img_size).to(device))
        optimizer = torch.optim.Adam([map], lr=1e-2)

        for ii in range(maxstep):
            map.data.clamp_(0, 0.5)
            P = map[:, 0:3, :, :]
            M = map[:, 3:6, :, :]
            rand = torch.rand_like(images).reshape(-1, 1)
            temp1 = torch.cat((rand, P.reshape(-1, 1), M.reshape(-1, 1)), 1)
            adv_img = Embbed(temp1).view(-1, 1, img_size, img_size)*1  + test_image
            
            adv_img = adv_img.view(-1,3,img_size,img_size)

            t_a = adv_img.view(-1,1,img_size,img_size)
            HPFa = HPF(t_a).reshape(-1,1,img_size,img_size)
            losshpf = criterionl1(cost * HPFa, cost * HPFtarget.detach())

            out = model(adv_img/255)
            _, pre = torch.max(out.data, 1)
            correct_train = (pre == labels).sum()
            # if pre.item() == labels.item():
            #     break
            CELOSS = CEL(out,labels)
            loss = CELOSS + c*losshpf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp = torch.sum(f(adv_img/255,model,labels)).item()
            # if ii % 100 == 0:
            #     print('step:{} order:{} loss:{} tmp:{} hpf:{}'.format(ii,order,CELOSS.item(),tmp,losshpf.item()))        
            if torch.sum(f(adv_img/255,model,labels)).item() > 0.001*batch_size:
                tmp = adv_img.clone().detach()
                tmp[tmp>255]=255
                tmp[tmp<0]=0
                tmp = (torch.round(tmp)).type(torch.uint8)
                outputs_white = model(tmp.type(torch.float)/255)
                _, pre_white = torch.max(outputs_white.data, 1)
                if pre_white == labels:
                    return adv_img
        return adv_img
    elif order == 2:
        target = images.view(-1,1,img_size,img_size)*255
        HPFtarget = HPF(target).reshape(-1,1,img_size,img_size)
        cost = HILLCOST(HPFtarget)
        cost[cost > 1] = 1
        cost = cost.detach()
        test_image = images.view(-1,1,img_size,img_size)*255
        map = Parameter(0.5 * torch.zeros(batch_size, 12, img_size, img_size).to(device))
        optimizer = torch.optim.Adam([map], lr=1e-2)
        for ii in range(maxstep):
            map.data.clamp_(0, 0.5)
            P = map[:, 0:3, :, :]
            M = map[:, 3:6, :, :]
            P_2 = map[:, 6:9, :, :]
            M_2 = map[:, 9:12, :, :]
            rand = torch.rand_like(images).reshape(-1, 1)
            rand2 = torch.rand_like(images).reshape(-1, 1)
            temp1 = torch.cat((rand, P.reshape(-1, 1), M.reshape(-1, 1)), 1)
            temp2 = torch.cat((rand2, P_2.reshape(-1, 1), M_2.reshape(-1, 1)), 1)
            adv_img = Embbed(temp1).view(-1, 1, img_size, img_size)*1 + Embbed(temp2).view(-1, 1, img_size, img_size)*2  + test_image
            
            adv_img = adv_img.view(-1,3,img_size,img_size)

            t_a = adv_img.view(-1,1,img_size,img_size)
            HPFa = HPF(t_a).reshape(-1,1,img_size,img_size)
            losshpf = criterionl1(cost * HPFa, cost * HPFtarget.detach())

            out = model(adv_img/255)
            _, pre = torch.max(out.data, 1)
            correct_train = (pre == labels).sum()
            # if pre.item() == labels.item():
            #     break
            CELOSS = CEL(out,labels)
            loss = CELOSS + c*losshpf
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp = torch.sum(f(adv_img/255,model,labels)).item()
            # if ii % 100 == 0:
            #     print('step:{} order:{} loss:{} tmp:{} hpf:{}'.format(ii,order,CELOSS.item(),tmp,losshpf.item()))        
            if torch.sum(f(adv_img/255,model,labels)).item() > 0.001*batch_size:
                tmp = adv_img.clone().detach()
                tmp[tmp>255]=255
                tmp[tmp<0]=0
                tmp = (torch.round(tmp)).type(torch.uint8)
                outputs_white = model(tmp.type(torch.float)/255)
                _, pre_white = torch.max(outputs_white.data, 1)
                if pre_white == labels:
                    return adv_img
        return adv_img

CEL = nn.CrossEntropyLoss()

total = 0
correct_white = 0
correct_black = 0
cost_hpf = 0
count_ori = start_num
count_adv = start_num
cost_l1 = 0
# torch.manual_seed(1) 
for images, labels in tqdm(normal_loader):
    labels = labels.to(device)
    images = images.to(device)

    labels = (torch.rand(batch_size))*1000
    labels = labels.to(device,dtype=torch.long)
    for order in range(1,4):
        adv_img = zn_attack(images,labels,order)
        adv_img[adv_img>255]=255
        adv_img[adv_img<0]=0
        images_adv = (torch.round(adv_img)).type(torch.uint8)
        outputs_white = model(images_adv.type(torch.float)/255)
        _, pre_white = torch.max(outputs_white.data, 1)
        if pre_white == labels:
            break
    images_adv = images_adv.cpu().numpy()
    images = (images*255).type(torch.uint8).cpu().numpy()
    saveimg(images_adv,'at')
    total += 1
    correct_white += (pre_white == labels).sum()
    cost_hpf += HPFLOSS(torch.from_numpy(images_adv.astype(np.float)).to(device=device,dtype=torch.float),torch.from_numpy(images.astype(np.float)).to(device=device,dtype=torch.float)).item()
    cost_l1 += np.sum(np.abs(images.astype(np.float)-images_adv.astype(np.float)))/images.size
    print('correct_white:{} order:{} cost_l1:{} cost_hpf:{}'.format(correct_white.item(),order,np.sum(np.abs(images.astype(np.float)-images_adv.astype(np.float)))/images.size,HPFLOSS(torch.from_numpy(images_adv.astype(np.float)).to(device=device,dtype=torch.float),torch.from_numpy(images.astype(np.float)).to(device=device,dtype=torch.float)).item()))

print(' Robust accuracy white: {:.5f}  costl1 :{:.5f} cost_hpf :{:.5f}'.format(
    (float(correct_white)/(batch_size*len(normal_loader))), cost_l1/len(normal_loader),cost_hpf/len(normal_loader)))
