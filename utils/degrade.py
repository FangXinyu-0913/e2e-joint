import torch
import torch.nn as nn
def postprocess(data): #数据恢复原有格式
    
    #三列，第一列为0-240，第二列为0-179，第三列为0或1
    #xs
    xs = data[:, 0]
    max_xs = xs.max().clone().detach() #这样使用解决
    min_xs = xs.min().clone().detach() 
    data[:, 0] = (xs - min_xs) / (max_xs - min_xs) * 239  #最大值不是240，是239.......
    #ys
    ys = data[:, 1]
    max_ys = ys.max().clone().detach()  #torch.max(ys)
    min_ys = ys.min().clone().detach()  #torch.min(ys) 
    data[:, 1] = (ys - min_ys) / (max_ys - min_ys) * 179    
    #pols维度,回归到0、1
    pols = data[:, 2]
    mean_pols = torch.mean(pols)
    
    #pols = [1 if x > mean_pols else 0 for x in pols ] 列表推导式，是python的处理,不可用
    pols = pols > mean_pols  #输出均为True，False这种,看结果确实是回到了0、1
    data[:, 2] = pols
    output = data
    return output
class fd_Net(nn.Module):
    def __init__(self):
        super(fd_Net,self).__init__()
        #1.是否会使用卷积层 2.fc层如何定义维度数量 3.此处只对于x,y,p做一个训练

        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64,3)
        self.relu = nn.ReLU(inplace=False)
        #self.dropout = nn.Dropout(0.5) 加dropout会报错
    def forward(self,input):

        input_t = input[:,2] #此列数据不变
        input_b = input[:,-1] #此列数据不变
        input_p = input[:,3]
        x = input[:,0:2]
        x = torch.cat([x,input_p.unsqueeze(1)],dim=1)
        x = self.fc1(x)
        # x = self.fc1(x.to(dtype=torch.float32))
        x1 = self.relu(x) #使用clone防止占位，之前是使用ReLU
        #x = self.dropout(x)
        x1 = self.fc2(x1)
        x1 = postprocess(x1)
        output_x_y = x1[:,0:2]
        output_p = x1[:,-1]
        output_1 = torch.cat([output_x_y,input_t.unsqueeze(1)],dim=1)
        output_2 = torch.cat([output_1,output_p.unsqueeze(1)],dim=1)
        output = torch.cat([output_2,input_b.unsqueeze(1)],dim=1)
        # print('第二列：','max:',input[:,1].max(),'min:',input[:,1].min())
        # print('第三列：','max:',input[:,2].max(),'min:',input[:,2].min())
        # print('第四列：','max:',input[:,3].max(),'min:',input[:,3].min())#,'all:',input[:,3])
        # print('output：')
        # print('第二列：','max:',output[:,1].max(),'min:',output[:,1].min())
        # print('第三列：','max:',output[:,2].max(),'min:',output[:,2].min())
        # print('第四列：','max:',output[:,3].max(),'min:',output[:,3].min())
       
        return output
