import  torch
import  numpy as np
import  torch.nn as nn
import  torch.optim as optim
from    matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR 

############################################CustomDataset_RNN################################################
class CustomDataset_RNN(Dataset):
    def __init__(self, file_path):
        self.samples = [] 
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = {}
        for line in lines:
            key, value = line.split(': ')
            data[key] = float(value)
            if (key=="msg.nohCtrlOutput.targetAcceleration"):
                self.samples.append(data.copy())
                data.clear()
    def __len__(self):
        return len(self.samples)-6# -6 : -1 is special for my case and -5 is sequence length
    def __getitem__(self, idx):
        sequence = self.samples[idx:idx+5] # e.g. 0 1 2 3 4 shape is (5,9)(squence length:5,features:9)
        features=[]
        # for i in range (5):
        #     sample=sequence[i]
        for sample in sequence:
            feature = torch.tensor([
            sample.get('msg.egoEgoStatus.yawRate', 0.0),  # set default value, in some case the dataset could be not printed correctly
            sample.get('msg.egoEgoStatus.linearSpeed', 0.0),
            sample.get('msg.egoEgoStatus.accerationX', 0.0),
            sample.get('msg.egoEgoStatus.accerationY', 0.0),
            sample.get('msg.egoEgoStatus.steerWheelAngle', 0.0),
            sample.get('msg.egoEgoStatus.steerWheelAngleRate', 0.0),
            sample.get('msg.egoEgoStatus.frontWheelAngle', 0.0),
            sample.get('msg.nohCtrlOutput.targetStrAngle', 0.0),
            sample.get('msg.nohCtrlOutput.targetAcceleration', 0.0)
            ])# dim 1
            features.append(feature)#dim 0 a list to tensor with shape (5,9)
        features = torch.stack(features, dim=0)
        sequence2 = self.samples[idx+6]# +1 is special process for my case
        labels=torch.tensor([
        sequence2.get('msg.egoEgoStatus.yawRate', 0.0),  
        sequence2.get('msg.egoEgoStatus.linearSpeed', 0.0),
        sequence2.get('msg.egoEgoStatus.accerationX', 0.0),
        sequence2.get('msg.egoEgoStatus.accerationY', 0.0),
        sequence2.get('msg.egoEgoStatus.steerWheelAngle', 0.0),
        sequence2.get('msg.egoEgoStatus.steerWheelAngleRate', 0.0),
        sequence2.get('msg.egoEgoStatus.frontWheelAngle', 0.0)
        ])
        labels=labels.view(1,-1)# or .unsqueeze(0) (7) to (1,7)
        return features,labels
############################################ my rnn################################################
class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(9,32)
        self.relu = nn.ReLU()
        self.linear_32=nn.Linear(32,32)
        self.rnn = nn.RNN(input_size=32, hidden_size=32, num_layers=1, batch_first=True) 
        self.linear_out=nn.Linear(32,7)
        # 创建 Batch Normalization 层
        self.batch_norm_layer = nn.BatchNorm1d(num_features=32,affine=True, track_running_stats=False)
        self.dropout = nn.Dropout(p=0.2)  # 添加 Dropout 层，丢弃率0.5
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)         
    def forward(self, x,hidden_prev):
        batch_size,sequence_length,num_features=x.shape
        temp1=self.linear_in(x)# temp1 (16,5,32)
        #temp1 = temp1.unsqueeze(2)  # 增加一个维度，变成 (batch_size, sequence_length, 1, input_size)
        temp1 = temp1.view(-1, 32)  # 将形状从 (batch_size, sequence_length, num_features) 改为 (batch_size*sequence_length, num_features) 
        temp11 = self.batch_norm_layer(temp1)
        temp11 = temp11.view(batch_size,sequence_length, 32)  # 再次变换形状回到原来的状态
        #temp1 = temp11.squeeze(2)  # 移除添加的维度
        temp2=self.relu (temp11)
        temp3=self.linear_32(temp2)# temp3 (16,5,32)
        out, hidden_prev = self.rnn(temp3,hidden_prev)# out (16,5,32) #hidden(1,16,32)        
        # Get output from the last time step
        last_output = out[:, -1, :]#last_output(16,32)   
        last_output = last_output.unsqueeze(1)# let (16,32)to(16,1,32)     
        # Feed into linear layer and apply activation
        #temp4 = self.linear_32(last_output)#temp4 (16,1,32)
        # #temp4 = temp4.unsqueeze(2)
        # temp4 = temp4.view(-1, 32) 
        # temp41 = self.batch_norm_layer(temp4)
        # temp4 = temp41.view(batch_size,1, 32)  
        #temp4 = temp41.squeeze(2)
 
        # 应用 Dropout 层
        temp5 = self.dropout(last_output)
        output =self.linear_out(temp5)#output(16,1,7)
        return output,hidden_prev
#####################training model define#################################
def train_RNN(dataloader,model,loss_fn,optimizer,device):
    #hidden_prev init
    size = len(dataloader.dataset)
    hidden_prev = None
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # print(X.shape, y.shape) 16 5 9 ;16 1 7
        # print (batch)
        X = X.to(device)  # 将数据移到设备上
        y = y.to(device)
        batch_size = X.size(0)
        hidden_prev = torch.zeros(1, batch_size, 32).to(device) if hidden_prev is None else hidden_prev
        # Detach hidden_prev from the computational graph
        hidden_prev = hidden_prev.detach() if hidden_prev is not None else None

        output, _ = model(X, hidden_prev)
        # hidden_prev = hidden_prev.detach()
        loss = loss_fn(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:# set print frequenz
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #l.append(loss.item())
#################################################test loop define ##########################################

def test_RNN(dataloader, model, loss_fn,device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    size = len(dataloader.dataset) # before batch operation
  
    num_batches = len(dataloader) #after batch operation
    #print(f"size: {size} and num_batches {num_batches}")
    test_loss, correct = 0, 0
    # # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
         for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)  # 将数据移到设备上
            y = y.to(device) 
            batch_size=X.size(0)
            hidden_prev = torch.zeros(1, batch_size, 32,dtype=X.dtype)
            hidden_prev = hidden_prev.to(device)  
            pred,_ = model(X,hidden_prev)
            loss_test=loss_fn(pred, y)
            test_loss += loss_test.item()
    #       #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if batch % 100 == 0:# set print frequenz
                loss, current = loss_test.item(), (batch + 1) * len(X)
                print(f"test loss: {loss_test:>7f}  [{current:>5d}/{size:>5d}]")

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
############################################main################################################



# main process        
if __name__ == "__main__":
    # build dataset 
    data_file_learning = 'training.txt' 
    data_file_t = 'test.txt' 
    custom_dataset_training = CustomDataset_RNN(data_file_learning)# (numbers,(features,labels)) features=(5,9) labels=(1,7)
    custom_dataset_valid = CustomDataset_RNN(data_file_t) 
    print(f"Number of samples in custom_dataset_training: {len(custom_dataset_training[0])}")# 2one sample: features and labels
    print(f"sequence length of features in one sample: {len(custom_dataset_training[0][0])}")# 5 sequence length of features
    print(f"sequence length of features in one sample: {len(custom_dataset_training[0][0][0])}")# 9 sequence length of features
    print(f"Number of features in one sample: {len(custom_dataset_training[0][1])}")# 1 sequence length of labels
    print(f"Number of features in one sample: {len(custom_dataset_training[0][1][0])}")# 7 sequence length of features

    # para
    learning_rate = 1e-3
    batch_size = 16
    epochs = 50
    # data loader
    #data_loader_training = DataLoader(custom_dataset_training, batch_size=batch_size, shuffle=True,drop_last=True)

    #data_loader_valid = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True, drop_last=True)

    #add device 
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # create model
    model =MyRNN().to(device) 
    
    print('model:\n',model)
    loss_fn = nn.MSELoss()  
    # create OP for BQ
    optimizer = optim.Adam(model.parameters(), learning_rate) 
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9) 
 # 将数据加载器移动到设备
    data_loader_training = DataLoader(custom_dataset_training, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    data_loader_valid = DataLoader(custom_dataset_valid, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    for data_batch, label_batch in data_loader_training:
        print(f"Shape of data_batch: {data_batch.shape}")#(16,5,9)

        for data, label in zip(data_batch, label_batch):
            print(f"Shape of data: {data.shape}")# (5,9)
            print(f"Shape of label: {label.shape} {label.dtype}")#(1,7)
            # for dat in data:
            #     print(f"data: {dat}")
            break
        break 
    # #hidden_prev init
    # hidden_prev = torch.zeros(1, 16, 32)
    for iter in range(epochs):
        print(f'_________________Epoch:{iter+1}/{epochs}_______________________')
        train_RNN(data_loader_training, model, loss_fn, optimizer, device)
        test_RNN(data_loader_valid, model, loss_fn, device)
        scheduler.step()
    # saving model 
    torch.save(model.state_dict(), 'VehicalStateperML_RNN.pth')
