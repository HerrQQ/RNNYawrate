import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=9, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        # # Reshape input to (batch_size, sequence_length, input_size)
        # x = x.unsqueeze(dim=0)
        
        # Pass through RNN layer
        out, _ = self.rnn(x)
        
        # Get output from the last time step
        last_output = out[:, -1, :]
        
        # Feed into linear layer and apply activation
        logits = self.linear(last_output)
        output = self.relu(logits)
        
        return output

class CustomDataset_Rnn(Dataset):
    def __init__(self, file_path):
        self.samples = []  

        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = {}
        for line in lines:

            #items = line.strip().split('\n')
            key, value = line.split(': ')
            data[key] = float(value)
            if (key=="msg.nohCtrlOutput.targetAcceleration"):
                self.samples.append(data.copy())
                data.clear()

    def __len__(self):
        return len(self.samples)-1# -1 is special process for my case

    def __getitem__(self, idx):
        sample = self.samples[idx]
        

        features = torch.tensor([
        sample.get('msg.egoEgoStatus.yawRate', 0.0),  # set default value, in some case the dataset could be not printed correctly
        sample.get('msg.egoEgoStatus.linearSpeed', 0.0),
        sample.get('msg.egoEgoStatus.accerationX', 0.0),
        sample.get('msg.egoEgoStatus.accerationY', 0.0),
        sample.get('msg.egoEgoStatus.steerWheelAngle', 0.0),
        sample.get('msg.egoEgoStatus.steerWheelAngleRate', 0.0),
        sample.get('msg.egoEgoStatus.frontWheelAngle', 0.0),
        sample.get('msg.nohCtrlOutput.targetStrAngle', 0.0),
        sample.get('msg.nohCtrlOutput.targetAcceleration', 0.0)
        ])
        sample2 = self.samples[idx+1]# +1 is special process for my case
        labels=torch.tensor([
        sample2.get('msg.egoEgoStatus.yawRate', 0.0),  
        sample2.get('msg.egoEgoStatus.linearSpeed', 0.0),
        sample2.get('msg.egoEgoStatus.accerationX', 0.0),
        sample2.get('msg.egoEgoStatus.accerationY', 0.0),
        sample2.get('msg.egoEgoStatus.steerWheelAngle', 0.0),
        sample2.get('msg.egoEgoStatus.steerWheelAngleRate', 0.0),
        sample2.get('msg.egoEgoStatus.frontWheelAngle', 0.0)
        ])

        return features,labels