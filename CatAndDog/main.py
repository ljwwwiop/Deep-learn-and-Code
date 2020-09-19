'''
    全部放在一起了,主要包括几个方面：
    1 训练: 定义网络，损失函数，优化器，进行训练，生成模型
    2 验证: 验证模型准确率
    3 测试: 测试模型在测试机上的准确率
    4 日志: 打印log的信息
'''

from config import opt
import os
import torch as t
from data.dataset import DogCat
from torch.utils.data import DataLoader
from utils.visualize import Visualizer
import torch.optim
from torch.autograd import Variable
from torchvision import models
from torch import nn
from torchnet import meter
import time
import csv


'''模型训练'''
def train(**kwargs):
    '''根据命令行参数更新配置'''
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    '''加载网络，若有预训练模型也加载'''
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512,2)
    # if opt.load_model_path:
    #	model.load(opt.load_model_path)
    if opt.use_gpu:  # GPU
        model.cuda()

    '''处理数据'''
    train_data = DogCat(opt.train_data_root,train=True) # 训练集
    val_data = DogCat(opt.train_data_root,train = False) # 验证集

    train_dataloader = DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)

    '''定义优化器，损失'''
    criterion = t.nn.CrossEntropyLoss() # 交叉熵
    lr = opt.lr
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr,weight_decay = opt.weight_decay)

    '''统计指标，平滑处理后的损失，还有混淆矩阵'''
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previos_loss = le10

    '''开始训练'''
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for i,(data,label) in enumerate(train_dataloader):
            print("i:",i)
            # 训练模型参数
            input = Variable(data)
            target = Variable(label)

            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            # 梯度归零
            optimizer.zero_grad()
            score = model(input)

            loss = criterion(score,target)
            loss.backward() # autograd 自动反向传播

            # 更新参数
            optimizer.step()
            # 更新统计指数指标及可视化
            loss_meter.add(loss.item())
            print(score.shape,target.shape)
            confusion_matrix.add(score.detach(), target.detach())

            if i % opt.print_freq == opt.print_freq-1:
                vis.plot("loss",loss_meter.value()[0])
        name = time.strftime('model' + '%m%d_%H:%M:%S.pth')
        t.save(model.state_dict(),'./checkpoint/'+name)

        '''验证'''
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".
                format(epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()),
                       train_cm=str(confusion_matrix.value()), lr=lr))
        print("loss_meter:",loss_meter)
        print("epoch:",epoch," loss",loss_meter.value()[0]," accuracy:",val_accuracy)

        '''如果损失不再下降，则降低学习率'''
        if loss_meter.value()[0] > previos_loss:
            lr = lr*opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previos_loss = loss_meter.value()[0]

"""计算模型在验证集上的准确率等信息"""
@t.no_grad()
def val(model,dataloader):
    model.eval() # 设置为验证模式

    confusion_matrix = meter.ConfusionMeter(2)
    for i,data in enumerate(dataloader):
        input,label = data
        val_input = Variable(input,volatile = True)
        val_label = Variable(label.long(),volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(),label.long())

    model.train() # 恢复为训练模式
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix

'''测试'''
def test(**kwargs):
    opt.parse(kwargs)

    # data
    test_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []

    # model ,如果有自己的模型了则可以直接 pretrained=False
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512,2)
    model.load_state_dict(t.load('./model.pth',map_location='cpu'))
    if opt.use_gpu:
        model.cuda()
    model.eval()

    for i,(data,path) in enumerate(test_dataloader):
        input = Variable(data,volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        path = path.numpy().tolist()
        _, predicted = t.max(score.data, 1)
        print("predicted",predicted)
        predicted = predicted.data.cpu().numpy().tolist()
        res = ""
        for (i,j) in zip(path,predicted):
            if j== 1:
                res ="dog"
            else:
                res = "cat"
            results.append([i,"".join(res)])
        print("results:",results)
    write_csv(results,opt.result_file)
    return results

'''写入csv'''
def write_csv(results,file_name):
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)
    f.close()

if __name__ == '__main__':
    import fire
    fire.Fire()



