import torch
# import scipy.io
from cfg import device



class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.fc1 = torch.nn.Linear(3,16)
        self.fc2 = torch.nn.Linear(16,32)
        self.fc3 = torch.nn.Linear(32, 1)

        self.relu = torch.nn.ReLU()
        self.sigomoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        out = self.tanh(x)
        return out
Embbed = TransformerNet()
Embbed.load_state_dict(torch.load('./step_fun.pkl'))
Embbed = Embbed.to(device)

class TransformerNet2(torch.nn.Module):
    def __init__(self):
        super(TransformerNet2, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.a = 10
    def forward(self,r,p):
        m = -0.5*self.tanh(self.a*(p-2*r))+0.5*self.tanh(self.a*(p-2*(1-r)))
        return m
Embbed2 = TransformerNet2()
Embbed2 = Embbed2.to(device)

if __name__=='__main__':
    # net = TransformerNet()
    # net.load_state_dict(torch.load('./step_fun.pkl'))
    # randChange = torch.rand(100,1)
    # t1 = torch.rand(100,1)/2
    # t2 = torch.rand(100,1)/2
    # input =torch.cat((randChange,t1,t2),1)
    # out = net(input)
    # tt = torch.cat((input,out),1)
    net = TransformerNet2()
    randChange = torch.rand(100,1)
    t1 = torch.rand(100,1)
    input =torch.cat((randChange,t1),1)
    out = net(randChange,t1)
    tt = torch.cat((input,out),1)
    print(tt)
    pass
# train_x = scipy.io.loadmat('./train_x.mat')
# train_x = train_x['t1']
# train_y = scipy.io.loadmat('./train_y.mat')
# train_y = train_y['y']
#
# train_x = torch.tensor(train_x, dtype=torch.float)
# train_y = torch.tensor(train_y, dtype=torch.float)
#
# net = TransformerNet().to(device)
#
# import torch.utils.data
# dataset = torch.utils.data.TensorDataset(train_x, train_y)
# train = torch.utils.data.DataLoader(dataset, batch_size=2000)
#
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#
# for i in range(1,501):
#     for x, y in train:
#         x = x.to(device)
#         y = y.to(device)
#         out = net(x)
#         loss = criterion(out, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#      # test_x = x[0,:,:,:].view(1,3,256,256)
#      # out_y = net(test_x)
#      # test_y = y[0,:,:,:].view(1,1,256,256)
#      # c = test_y - out_y
#     print('epoch:{} loss:{} '.format(i,loss.item()))
# torch.save(net.state_dict(), 'step_fun.pkl')
# #
# import torch
# import scipy.io
# test_net = TransformerNet()
# test_net.load_state_dict(torch.load('./step_fun.pkl'))
#
# test_x = scipy.io.loadmat('./test_x.mat')
# test_x = torch.tensor(test_x['t1'], dtype = torch.float).to(device)
# test_y = scipy.io.loadmat('./test_y.mat')
# test_y = torch.tensor(test_y['y'], dtype = torch.float).to(device)
# out = net(test_x)
# out_n = out.detach().cpu().numpy()
# c = out - test_y
# test_y_n = test_y.detach().cpu().numpy()
# c = c.detach().cpu().numpy()
