import torch
from torch import nn
import os

from rnnForYawrate import MyRNN
from rnnForYawrate import CustomDataset_RNN
from torch.utils.data import Dataset, DataLoader
# main process        
if __name__ == "__main__":

    # current_path = os.getcwd()
    # print("当前工作路径:", current_path)
    # new_path = current_path
    # os.chdir(new_path)

    data_file = 'validationData.txt'  
    custom_dataset_inference = CustomDataset_RNN(data_file)
    batch_size=16
    # 数据加载器
    data_loader_inference = DataLoader(custom_dataset_inference, batch_size=batch_size,shuffle=True, drop_last=True)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model_f=MyRNN().to(device)
    model_f.load_state_dict(torch.load('VehicalStateperML_RNN.pth'))
    model_f.eval()

    if 1:
    # single data inference    
        #hidden_prev init
        hidden_prev = torch.zeros(1, 1, 32) #
        
        data_test_1=torch.tensor([0.03987893462181091,13.782072067260742,0.21904867887496948,0.8140702247619629,0.14955520629882812,
        0.0,0.01074754074215889,8.803881645202637,-0.25467321276664734])
        
        data_test_2=torch.tensor([0.038624390959739685,13.786596298217773,0.2508492171764374,0.8024824857711792,0.15127170085906982,
        0.0,0.010829867795109749,8.872920989990234,-0.3033924400806427])
        
        data_test_3=torch.tensor([0.03859284520149231,13.785704612731934,0.15484023094177246,0.8001524209976196,0.1524304449558258,
        0.0,0.010974112898111343,8.793074607849121,-0.34160029888153076])

        data_test_4=torch.tensor([0.03933088481426239,13.788601875305176,0.2645440995693207,0.8018520474433899,0.15475332736968994,
        0.0,0.01099490374326706,8.803157806396484,-0.4666106402873993])

        data_test_5=torch.tensor([0.039885520935058594,13.795905113220215,0.16176363825798035,0.7602496147155762,0.15518921613693237,
        0.0,0.011025872081518173,8.80613899230957,-0.5674195289611816])
        stacked_data = torch.stack((data_test_1, data_test_2, data_test_3, data_test_4, data_test_5)) #shape (5,9)
        stacked_data=stacked_data.unsqueeze(0)# shape (1,5,9)
        with torch.no_grad():
            stacked_data = stacked_data.to(device)  
            hidden_prev =hidden_prev.to(device)
            output,_ = model_f(stacked_data,hidden_prev)  # 使用模型进行预测
        print(output) 
    # else:
    # nulti data inference model

        # loss_fn = nn.MSELoss()
        # trainingModel.test_loop(data_loader_inference, model_f, loss_fn)

# RNN single data inference model
# msg.egoEgoStatus.yawRate: 0.03987893462181091
# msg.egoEgoStatus.linearSpeed: 13.782072067260742
# msg.egoEgoStatus.accerationX: 0.21904867887496948
# msg.egoEgoStatus.accerationY: 0.8140702247619629
# msg.egoEgoStatus.steerWheelAngle: 0.14955520629882812
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.010625587776303291
# msg.nohCtrlOutput.targetStrAngle: 8.803881645202637
# msg.nohCtrlOutput.targetAcceleration: -0.25467321276664734
# msg.egoEgoStatus.yawRate: 0.038624390959739685
# msg.egoEgoStatus.linearSpeed: 13.786596298217773
# msg.egoEgoStatus.accerationX: 0.23861587047576904
# msg.egoEgoStatus.accerationY: 0.8024824857711792
# msg.egoEgoStatus.steerWheelAngle: 0.15127170085906982
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.01074754074215889
# msg.nohCtrlOutput.targetStrAngle: 8.8500394821167
# msg.nohCtrlOutput.targetAcceleration: -0.3033924400806427
# msg.egoEgoStatus.yawRate: 0.03859284520149231
# msg.egoEgoStatus.linearSpeed: 13.785704612731934
# msg.egoEgoStatus.accerationX: 0.2508492171764374
# msg.egoEgoStatus.accerationY: 0.8001524209976196
# msg.egoEgoStatus.steerWheelAngle: 0.1524304449558258
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.010829867795109749
# msg.nohCtrlOutput.targetStrAngle: 8.872920989990234
# msg.nohCtrlOutput.targetAcceleration: -0.34160029888153076
# msg.egoEgoStatus.yawRate: 0.03933088481426239
# msg.egoEgoStatus.linearSpeed: 13.788601875305176
# msg.egoEgoStatus.accerationX: 0.2645440995693207
# msg.egoEgoStatus.accerationY: 0.8018520474433899
# msg.egoEgoStatus.steerWheelAngle: 0.15475332736968994
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.01099490374326706
# msg.nohCtrlOutput.targetStrAngle: 8.803157806396484
# msg.nohCtrlOutput.targetAcceleration: -0.4666106402873993
# msg.egoEgoStatus.yawRate: 0.039885520935058594
# msg.egoEgoStatus.linearSpeed: 13.795905113220215
# msg.egoEgoStatus.accerationX: 0.16176363825798035
# msg.egoEgoStatus.accerationY: 0.7602496147155762
# msg.egoEgoStatus.steerWheelAngle: 0.15518921613693237
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.011025872081518173
# msg.nohCtrlOutput.targetStrAngle: 8.80613899230957
# msg.nohCtrlOutput.targetAcceleration: -0.5674195289611816

#ans:
# msg.egoEgoStatus.yawRate: 0.04009075462818146
# msg.egoEgoStatus.linearSpeed: 13.800976753234863
# msg.egoEgoStatus.accerationX: 0.20156517624855042
# msg.egoEgoStatus.accerationY: 0.7818496227264404
# msg.egoEgoStatus.steerWheelAngle: 0.15529820322990417
# msg.egoEgoStatus.steerWheelAngleRate: 0.0
# msg.egoEgoStatus.frontWheelAngle: 0.011033616028726101



