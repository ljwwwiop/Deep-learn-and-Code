'''
    配置文件：将一些所有可以配置的参数都放在这个文件中
    参数成员：
    数据集参数：batch_size,文件路径
    训练参数: 超参数，epoch,lr..
'''
import warnings
import torch as t

class DefaultConfig(object):

    #  visdom环境和端口
    env = 'default'
    vis_port = 8097

    model = "ResNet34" # 选择模型


    # train_data_root = "/home/gzshan/sgz/dogsVScats/data/catvsdog/train/"  # 训练集存放路径
    test_data_root = "./data/test/"  # 测试集存放路径

    load_model_path = None  # 加载预训练模型的路径，None表示不加载

    # 训练参数
    batch_size = 8  # batch_size
    use_gpu = False
    num_workers = 4
    print_freq = 50

    # 文件路径
    debug_file = ""
    result_file = "./submission.csv"

    # 模型参数
    max_epoch = 10
    lr = 0.001
    lr_decay = 0.95
    weight_decay = 1e-4  # 损失函数

    """根据字典更新config参数，便于命令行更改参数"""
    def parse(self,kwargs):
        '''更新配置参数'''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning:opt has not attribut %s" % k)
            setattr(self,k,v)

        '''打印配置信息'''
        print("user config:")
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))

opt = DefaultConfig()



