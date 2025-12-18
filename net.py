# # 注意力机制+极限学习机网络
# # from easydl import *
# from torchvision import models
# from data import *
# from torch import nn
# import torch
#
# class BaseFeatureExtractor(nn.Module):  # 特征提取器的基准类，作为不同特征提取器的父类
#     def forward(self, *input):          # 在基类中定义了前向传播的方法，在调用的时候便于改写
#         pass                            # 定义基类的好处是，可以确保不同网络框架下特征提取器的基本内容保持不变
#
#     def __init__(self):
#         super(BaseFeatureExtractor, self).__init__()
#
#     def output_num(self):               #输出方法，输出特征向量
#         pass
#
#     def train(self, mode=True):         #不懂？？？？？？？？？？？？？？？？？？
#         # freeze BN mean and std
#         for module in self.children():
#             if isinstance(module, nn.BatchNorm2d):  #isinstance对比两个参数是否相同
#                 module.train(False)
#             else:
#                 module.train(mode)
#
# #VGG和resnet的存在是为了做对比
# class ResNet50Fc(BaseFeatureExtractor):  #resnet50特征提取器，输入图像需要进行归一化处理
#     """
#     ** input image should be in range of [0, 1]**
#     """
#     def __init__(self,model_path=None, normalize=True):
#         super(ResNet50Fc, self).__init__()
#         if model_path:
#             if os.path.exists(model_path):
#                 self.model_resnet = models.resnet50(pretrained=False)     # 作者不使用pytoch提供的预训练过的模型，用一个新模型
#                 self.model_resnet.load_state_dict(torch.load(model_path))  # 加载自己训练好的模型参数
#             else:
#                 raise Exception('invalid model path!')      # 如果找不到有效文件就会报错
#         else:
#             self.model_resnet = models.resnet50(pretrained=True)  #如果没有模型参数，就直接使用pytorch提供的在Image net上预训练好的模型
#
#         if model_path or normalize:
#             # pretrain model is used, use ImageNet normalization
#             self.normalize = True
#             self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  #缓冲区可以使用给定的名称作为属性访问 比如mean
#             self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))   #三个数值，R B G ，三个通道的均值和方差
#         else:
#             self.normalize = False
#
#         model_resnet = self.model_resnet
#         self.conv1 = model_resnet.conv1
#         self.bn1 = model_resnet.bn1
#         self.relu = model_resnet.relu
#         self.maxpool = model_resnet.maxpool
#
#         self.layer1 = model_resnet.layer1
#         self.layer2 = model_resnet.layer2
#         self.layer3 = model_resnet.layer3
#         self.layer4 = model_resnet.layer4
#         self.avgpool = model_resnet.avgpool
#         self.__in_features = model_resnet.fc.in_features
#
#     def forward(self, x):
#         if self.normalize:
#             x = (x - self.mean) / self.std
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)                 #最大池化
#         x = self.layer1(x)                  #第一个残差模块
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)           # x = x.view(x.size(0), -1)  这句话的出现就是为了将前面多维度的tensor展平成一维
#         # print("resnet_output_size:",x.shape)
#         return x                            #简化x = x.view(batchsize, -1) 这里才是特征向量吧？
#
#     def output_num(self):
#         return self.__in_features
#
#
# class VGG16Fc(BaseFeatureExtractor):
#     def __init__(self,model_path=None, normalize=True):
#         super(VGG16Fc, self).__init__()
#         if model_path:
#             if os.path.exists(model_path):
#                 self.model_vgg = models.vgg16(pretrained=False)
#                 self.model_vgg.load_state_dict(torch.load(model_path))
#             else:
#                 raise Exception('invalid model path!')
#         else:
#             self.model_vgg = models.vgg16(pretrained=True)
#
#         if model_path or normalize:
#             # pretrain model is used, use ImageNet normalization
#             self.normalize = True
#             self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#             self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
#         else:
#             self.normalize = False
#
#         model_vgg = self.model_vgg
#         self.features = model_vgg.features
#         self.classifier = nn.Sequential()               #初始化分类器网络
#         for i in range(6):                              #model_vgg.classifier[i]导入的是VGG16的分类器，但是不导入最后一个分类层
#             self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])  #添加则模块，如果网络中出现了相同名称的子模块就变成替换行为了
#         self.feature_layers = nn.Sequential(self.features, self.classifier)
#
#         self.__in_features = 4096   #分类器网络中最后一层的输入个数
#
#     def forward(self, x):
#         if self.normalize:
#             x = (x - self.mean) / self.std
#         x = self.features(x)    #用的是VGG的特征提取网络模块
#         x = x.view(x.size(0), 25088)    #数据平铺
#         x = self.classifier(x) #输出特征向量
#         return x
#
#     def output_num(self):
#         return self.__in_features
#
# #分类器，供应模型框架中的分类器和辅助分类器部分
# class CLS(nn.Module):
#     """
#     a two-layer MLP for classification
#     """
#     def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
#         super(CLS, self).__init__()
#         self.pretrain = pretrain
#         if bottle_neck_dim:  # 感觉这是区分两个分类器的关键  源域分类器
#             self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
#             self.fc = nn.Linear(bottle_neck_dim, out_dim)
#             self.main = nn.Sequential(self.bottleneck,self.fc,nn.Softmax(dim=-1))  # 对分类器网络进行了组合
#         else:
#             self.fc = nn.Linear(in_dim, out_dim)
#             self.main = nn.Sequential(self.fc,nn.Softmax(dim=-1))
#
#     def forward(self, x):
#         out = [x]  # out序列的第二个值是维度为256，相当于把输入的特征向量再进行过一次处理后维度变成256维
#         for module in self.main.children():  # 迭代main里面的子模块，它不会访问到子模块的内部中
#             x = module(x)
#             out.append(x)    # out列表包含每一个模块的输出，第一个是x本身也就是输入的特征向量，最后一个是分类，中间的序列值是全连接层的生成值
#         return out
#
# # 带有梯度反转层的对抗网络
#
# # 对抗网络
# class AdversarialNetwork(nn.Module):
#     """
#     AdversarialNetwork with a gredient reverse layer.
#     its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
#     """
#     def __init__(self, in_feature):
#         super(AdversarialNetwork, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(in_feature, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(1024,1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 1),
#             nn.Sigmoid()   #这里有个sigmoid，二分类，可用BCEloss
#         )
#         self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))
#
#     def forward(self, x):
#         x_ = self.grl(x)            #梯度反转层 不明白！！？？？？？？？
#         y = self.main(x_)
#         return y
#
# # print("****************net加载结束********************")
# # 添加简单的极限学习机作为分类器或者域判别器
# import numpy as np
# #
# # #把极限学习机作为辅助分类器，仅需要输入源域数据和标签即可，最后输出的值主要是给源域加权用
# # #标签改为onehot比较好
#
#
#
# class Manifold(nn.Module):  # 低维黎曼流行嵌入与对齐层，由三层线性网络组成
#     def __init__(self, input_dim, conv_dim_1, conv_dim_2, n_classes):
#         super(Manifold, self).__init__()
#         self.fc1 = nn.Sequential(
#               nn.Linear(input_dim, conv_dim_1),
#               nn.BatchNorm1d(conv_dim_1),
#               nn.LeakyReLU(negative_slope=0.2, inplace=True),
#              )
#         self.fc2 = nn.Sequential(
#               nn.Linear(conv_dim_1, conv_dim_2),
#               nn.BatchNorm1d(conv_dim_2),
#               nn.Tanh(),
#              )
#         self.fc3 = nn.Sequential(      # 考虑要不要这部分
#               nn.Linear(conv_dim_2, n_classes),
#               nn.Softmax(),
#              )
#
#     def forward(self, x):
#         z_1 = self.fc1(x)  # 第一层嵌入流行输出
#         z_2 = self.fc2(z_1)  # 第二层嵌入流行输出
#         y = self.fc3(z_2)  #
#         return y, z_1, z_2
#
# #       Source Inter-class Similarity
# # ==========================================================
#
#
# def Source_InterClass_sim_loss(h_s, target, source_Tmean, Sim_type='sum'):
#     uni_tar = target.unique()  # target是标签,source_Tmean源域中心？
#     # print("target.type:",target.type)
#     num_sam_class = torch.zeros(uni_tar.shape[0]).cuda()
#     # print('h_s:', h_s)
#     Class_mean = torch.zeros(uni_tar.shape[0], h_s.shape[1]).cuda()
#
#     # print('Class_mean:', Class_mean)
#     for i in range(uni_tar.shape[0]):
#         Index_i = (target == uni_tar[i])
#         num_sam_class[i] = Index_i.sum()
#         Class_mean[i, :] = h_s[Index_i, :].mean(0)
#         # print(i)
#         # print('h_s[Index_i, :].mean(0):',h_s[Index_i, :].mean(0))
#     # print('Class_mean:', Class_mean)
#     Class_mean = Class_mean - source_Tmean.repeat(Class_mean.shape[0], 1)
#     # print('Class_mean:', Class_mean)
#     # print('source_Tmean:',source_Tmean)
#     norm_CM = Class_mean.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
#     Class_mean = Class_mean.mul(1 / (norm_CM + 1e-8))
#     # print('norm_CM:',norm_CM)
#
#     SIM = 0
#
#     if Sim_type == 'adj':
#         for i in range(uni_tar.shape[0] - 1):
#             for j in range(i + 1, uni_tar.shape[0]):
#                 SIM += (Class_mean[i].mul(Class_mean[j]).sum() / 2 + 1 / 2).pow(2)
#         # print('(target.shape[0] * (target.shape[0] - 1) / 2):',(target.shape[0] * (target.shape[0] - 1) / 2))
#         return SIM / (target.shape[0] * (target.shape[0] - 1) / 2)
#     elif Sim_type == 'none':
#         for i in range(uni_tar.shape[0] - 1):
#             for j in range(i + 1, uni_tar.shape[0]):
#                 SIM += Class_mean[i].mul(Class_mean[j]).sum().pow(2)
#         return SIM / (target.shape[0] * (target.shape[0] - 1) / 2)
#     elif Sim_type == 'sum':
#         for i in range(uni_tar.shape[0] - 1):
#             for j in range(i + 1, uni_tar.shape[0]):
#                 SIM += Class_mean[i].mul(Class_mean[j]).sum() + 1 / 2
#         return SIM / (target.shape[0] * (target.shape[0] - 1) / 2)
#
#
# # ==========================================================
# #       Target Intra-class Similarity
# # ==========================================================
#
#
# def Target_IntraClass_sim_loss(h_t, dim, source_mean, source_lab):  # h_t表示目标域在流行上的特征矩阵。 source_mean表示源域类中心
#     s_mean = torch.ones(len(source_lab),dim).cuda()
#     num = 0
#     for i in source_lab:
#         i = i.item()
#         s_mean[num] = source_mean[i].cuda()
#         num += 1
#
#     norm_h = h_t.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
#     norm_s = s_mean.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
#     h_t = h_t.mul(1 /(norm_h + 1e-8))  # 对特征矩阵的数据进行了处理
#     s_mean = s_mean.mul(1 / (norm_s + 1e-8))  # 对源域类中心数据进行了处理
#     Flag = torch.ones(s_mean.shape).cuda()
#     sim1 = torch.zeros((len(source_lab),1)).cuda()
#     num = 0
#     for i in s_mean:
#         Flag = Flag*i
#         sim = h_t.mm(Flag.t())
#         c = sim.norm('fro').pow(2) / 12
#         Flag = torch.ones(s_mean.shape).cuda()
#         # a=torch.min(c)
#         # sim1[num] = 1/(c + 1e-10)
#         sim1[num] = c
#
#         num +=1
#     # Flag = torch.ones(pred_t.shape).cuda()
#     # if Top_n:  # 截断操作
#     #     _, De_index = pred_t.sort(1, descending=True)  # 预测数据的排序
#     #     for i in range(pred_t.shape[0]):
#     #         Flag[i, De_index[i, Top_n:]] = 0
#     # sim1 = h_t.mm(s_mean.t())
#     # a = sim1.norm('fro').pow(2) # 输出一个值
#     # b = sim1.norm('fro').pow(2) / (pred_t.shape[0] * pred_t.shape[1]) # 输出一个值
#     # c = sim1.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
#     # 判断一下输出维度
#     # sim1 = sim1 / (torch.max(sim1) + 1e-10)
#     return sim1
#
#     # Sim = sim1.mul(pred_t).mul(Flag)
#     # Square F-norm of weighted intra-class scatter
#     # return -Sim.norm('fro').pow(2) / (pred_t.shape[0] * pred_t.shape[1])
#
# # =================================================================
#
#
# def weights_init(m):   # 初始化权值，是用来初始化嵌入流行层的
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.1)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0.1)
#     elif classname.find('Linear') != -1:
#         m.weight.data.normal_(0.0, 0.1)
#         m.bias.data.fill_(0.1)
#
#
# # ==========================================================
# #       Grassmannian Manifold Metric
# # ==========================================================
#
#
# def grassmann_dist_Fast(input1, input2,source_lab,dim):  # 这里的输入是源域和目标域的协方差矩阵
#     s_mean = torch.zeros(len(source_lab), dim).cuda()
#     # weights = torch.zeros(len(source_lab)).cuda()
#     num = 0
#     for i in source_lab:
#         i = i.item()
#         s_mean[num] = input1[i].cuda()
#         num += 1
#     # min_dis = 10
#     # for i in s_mean:
#     #     num = 0
#     #     for j in input2:
#     #         # fea_dim = i.shape[1]
#     #
#     #         h_src = i - torch.mean(input1, dim=0)  # 减去均值，去中心化
#     #         h_trg = j - torch.mean(input2, dim=0)
#     #
#     #         _, D1, V1 = torch.svd(h_src)  # 协方差矩阵的SVD分解 SVD分解至少两维
#     #         _, D2, V2 = torch.svd(h_trg)
#     #
#     #         grassmann_dist = torch.sum(torch.pow(V1.mm(V1.t()) - V2.mm(V2.t()), 2))
#     #         if grassmann_dist < min_dis :
#     #             min_dis = grassmann_dist
#     #     weights[num] = min_dis
#     #     num +=1
#     input1 = s_mean
#     fea_dim = input1.shape[1]
#
#     h_src = input1 - torch.mean(input1, dim=0)  # 减去均值，去中心化
#     h_trg = input2 - torch.mean(input2, dim=0)
#
#     _, D1, V1 = torch.svd(h_src)  # 协方差矩阵的SVD分解
#     _, D2, V2 = torch.svd(h_trg)
#
#     grassmann_dist = torch.sum(torch.pow(V1.mm(V1.t())-V2.mm(V2.t()), 2))/(fea_dim*fea_dim)
#
#     return grassmann_dist
#
# #==========================================================
#
# # ==========================================================
# #       Compute Accuracy
# # ==========================================================
#
#
# def classification_accuracy(e, c, data_loader):
#     with torch.no_grad():
#         correct = 0
#         for batch_idx, (X, target) in enumerate(data_loader):
#             X, target = Variable(X), Variable(target).long().squeeze()
#             X, target = X.cuda(), target.cuda()
#             t_fe = e.forward(X)
#
#             output, _, _ = c(t_fe)
#
#             pred = output.data.max(1)[1]
#             correct += pred.eq(target.data).cpu().sum()
#
#         return correct.item() / len(data_loader.dataset)
#
#
# 注意力机制+极限学习机网络
# from easydl import *
from torchvision import models
from data import *
from torch import nn
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class BaseFeatureExtractor(nn.Module):  # 特征提取器的基准类，作为不同特征提取器的父类
    def forward(self, *input):          # 在基类中定义了前向传播的方法，在调用的时候便于改写
        pass                            # 定义基类的好处是，可以确保不同网络框架下特征提取器的基本内容保持不变

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):               #输出方法，输出特征向量
        pass

    def train(self, mode=True):         #不懂？？？？？？？？？？？？？？？？？？
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):  #isinstance对比两个参数是否相同
                module.train(False)
            else:
                module.train(mode)

#VGG和resnet的存在是为了做对比
class ResNet50Fc(BaseFeatureExtractor):  #resnet50特征提取器，输入图像需要进行归一化处理
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)     # 作者不使用pytoch提供的预训练过的模型，用一个新模型
                self.model_resnet.load_state_dict(torch.load(model_path))  # 加载自己训练好的模型参数
            else:
                raise Exception('invalid model path!')      # 如果找不到有效文件就会报错
        else:
            self.model_resnet = models.resnet50(pretrained=True)  #如果没有模型参数，就直接使用pytorch提供的在Image net上预训练好的模型

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  #缓冲区可以使用给定的名称作为属性访问 比如mean
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))   #三个数值，R B G ，三个通道的均值和方差
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool

        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)                 #最大池化
        x = self.layer1(x)                  #第一个残差模块
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)           # x = x.view(x.size(0), -1)  这句话的出现就是为了将前面多维度的tensor展平成一维
        # print("resnet_output_size:",x.shape)
        return x                            #简化x = x.view(batchsize, -1) 这里才是特征向量吧？

    def output_num(self):
        return self.__in_features

class AlexNet(BaseFeatureExtractor):
    def __init__(self,model_path=None):
        super(AlexNet, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_alexnet = models.alexnet(pretrained=False)     # 作者不使用pytoch提供的预训练过的模型，用一个新模型
                self.model_alexnet.load_state_dict(torch.load(model_path))  # 加载自己训练好的模型参数
            else:
                raise Exception('invalid model path!')      # 如果找不到有效文件就会报错
        else:
            self.model_alexnet = models.alexnet(pretrained=True)  #如果没有模型参数，就直接使用pytorch提供的在Image net上预训练好的模型
        self.features = self.model_alexnet.features
        # self.pool = self.model_alexnet.avgpool


    def forward(self, x):
        x = self.features(x)
        # x = self.pool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        return x
    def output_num(self):
        return 256 * 6 * 6

class AlexNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(AlexNetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(4096, bottleneck_dim)
            self.bottleneck.weight.data.normal_(0, 0.005)
            self.bottleneck.bias.data.fill_(0.0)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(4096, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
            self.__in_features = 4096
    else:
        self.fc = model_alexnet.classifier[6]
        self.__in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x

  def output_num(self):
    return self.__in_features


class VGG16Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(VGG16Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_vgg = models.vgg16(pretrained=False)
                self.model_vgg.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_vgg = models.vgg16(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_vgg = self.model_vgg
        self.features = model_vgg.features
        self.classifier = nn.Sequential()               #初始化分类器网络
        for i in range(6):                              #model_vgg.classifier[i]导入的是VGG16的分类器，但是不导入最后一个分类层
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])  #添加则模块，如果网络中出现了相同名称的子模块就变成替换行为了
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.__in_features = 4096   #分类器网络中最后一层的输入个数

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)    #用的是VGG的特征提取网络模块
        x = x.view(x.size(0), 25088)    #数据平铺
        x = self.classifier(x) #输出特征向量
        return x

    def output_num(self):
        return self.__in_features

#分类器，供应模型框架中的分类器和辅助分类器部分
class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
        super(CLS, self).__init__()
        self.pretrain = pretrain
        if bottle_neck_dim:  # 感觉这是区分两个分类器的关键  源域分类器
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(self.bottleneck,self.fc,nn.Softmax(dim=-1))  # 对分类器网络进行了组合
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(self.fc,nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]  # out序列的第二个值是维度为256，相当于把输入的特征向量再进行过一次处理后维度变成256维
        for module in self.main.children():  # 迭代main里面的子模块，它不会访问到子模块的内部中
            x = module(x)
            out.append(x)    # out列表包含每一个模块的输出，第一个是x本身也就是输入的特征向量，最后一个是分类，中间的序列值是全连接层的生成值
        return out


class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """

    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
        super(CLS, self).__init__()
        self.pretrain = pretrain

        if bottle_neck_dim:
            # Update bottleneck layer's input feature dimension
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)

            # Update fc layer's input from bottleneck's output feature dimension
            self.fc = nn.Linear(bottle_neck_dim, out_dim)

            # Update the main sequence with the modified bottleneck and fc layers
            self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))
        else:
            # No changes here as there's no bottleneck layer
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


# 带有梯度反转层的对抗网络

# 对抗网络
class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()   #这里有个sigmoid，二分类，可用BCEloss
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)            #梯度反转层 不明白！！？？？？？？？
        y = self.main(x_)
        return y

# print("****************net加载结束********************")
# 添加简单的极限学习机作为分类器或者域判别器
import numpy as np
#
# #把极限学习机作为辅助分类器，仅需要输入源域数据和标签即可，最后输出的值主要是给源域加权用
# #标签改为onehot比较好



class Manifold(nn.Module):  # 低维黎曼流行嵌入与对齐层，由三层线性网络组成
    def __init__(self, input_dim, conv_dim_1, conv_dim_2, n_classes):
        super(Manifold, self).__init__()
        self.fc1 = nn.Sequential(
              nn.Linear(input_dim, conv_dim_1),
              nn.BatchNorm1d(conv_dim_1),
              nn.LeakyReLU(negative_slope=0.2, inplace=True),
             )
        self.fc2 = nn.Sequential(
              nn.Linear(conv_dim_1, conv_dim_2),
              nn.BatchNorm1d(conv_dim_2),
              nn.Tanh(),
             )
        self.fc3 = nn.Sequential(      # 考虑要不要这部分
              nn.Linear(conv_dim_2, n_classes),
              nn.Softmax(),
             )

    def forward(self, x):
        z_1 = self.fc1(x)  # 第一层嵌入流行输出
        z_2 = self.fc2(z_1)  # 第二层嵌入流行输出
        y = self.fc3(z_2)  #
        return y, z_1, z_2

#       Source Inter-class Similarity
# ==========================================================
def Source_InterClass_sim(h_s, target, source_Tmean, Sim_type='adj'):
    uni_tar = target.unique()  # target是标签,source_Tmean源域中心？
    # print("target.type:",target.type)
    num_sam_class = torch.zeros(uni_tar.shape[0]).cuda()
    Class_mean = torch.zeros(uni_tar.shape[0], h_s.shape[1]).cuda()
    for i in range(uni_tar.shape[0]):
        Index_i = (target == uni_tar[i])
        num_sam_class[i] = Index_i.sum()
        Class_mean[i, :] = h_s[Index_i, :].mean(0)

    Class_mean = Class_mean - source_Tmean.repeat(Class_mean.shape[0], 1)
    norm_CM = Class_mean.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
    Class_mean = Class_mean.mul(1 / norm_CM)

    SIM = 0

    if Sim_type == 'adj':
        for i in range(uni_tar.shape[0] - 1):
            for j in range(i + 1, uni_tar.shape[0]):
                SIM += (Class_mean[i].mul(Class_mean[j]).sum() / 2 + 1 / 2).pow(2)
        return SIM / (target.shape[0] * (target.shape[0] - 1) / 2)
    elif Sim_type == 'none':
        for i in range(uni_tar.shape[0] - 1):
            for j in range(i + 1, uni_tar.shape[0]):
                SIM += Class_mean[i].mul(Class_mean[j]).sum().pow(2)
        return SIM / (target.shape[0] * (target.shape[0] - 1) / 2)
    elif Sim_type == 'sum':
        for i in range(uni_tar.shape[0] - 1):
            for j in range(i + 1, uni_tar.shape[0]):
                SIM += Class_mean[i].mul(Class_mean[j]).sum() + 1 / 2
        return SIM / (target.shape[0] * (target.shape[0] - 1) / 2)

# def Source_InterClass_sim(h_s, target, target_Tmean):
#     '''
#     PLGE_inter_loss_L2 = Source_InterClass_sim_loss(feature_source, label_source, target_Tmean,'adj')
#     '''
#     Flag = torch.ones(h_s.shape).cuda()
#     Flag = Flag*target_Tmean
#     a=h_s.mm(Flag.t())
#     sim = torch.sum(h_s.mm(Flag.t()), 1) / (h_s.shape[1]*h_s.shape[0])
#
#     return sim/torch.max(sim)


# ==========================================================
#       Target Intra-class Similarity
# ==========================================================


def Target_IntraClass_sim(h_t,h_s, dim, source_mean,target_Tmean, source_lab):  # h_t表示目标域在流行上的特征矩阵。 source_mean表示源域类中心
    '''
    PLGE_loss_L2 = Target_IntraClass_sim_loss(feature_target, 256, source_class_means, label_source)
    torch.mul(a, b) 是矩阵a和b对应位相乘，a和b的维度必须相等。
    torch.mm(a, b) 是矩阵a和b矩阵相乘
    '''
    s_mean = torch.ones(len(source_lab),dim).cuda()
    num = 0
    for i in source_lab:
        i = i.item()
        s_mean[num] = source_mean[i].cuda()
        num += 1

    norm_h = h_t.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
    norm_s = s_mean.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
    h_t = h_t.mul(1 /(norm_h + 1e-8))  # 对特征矩阵的数据进行了处理
    s_mean = s_mean.mul(1 / (norm_s + 1e-8))  # 对源域类中心数据进行了处理

    Flag = torch.ones(s_mean.shape).cuda()
    sim1 = torch.zeros((len(source_lab),1)).cuda()
    num = 0
    for i in s_mean:
        Flag = Flag*i
        # b=h_t.mm(Flag.t())
        sim = torch.sum(h_t.mm(Flag.t()),1)/(h_t.shape[0])

        Flag = torch.ones(s_mean.shape).cuda()
        sim1[num] = max(sim)
        num +=1
    sim1 = sim1 / (torch.mean(sim1, dim=0, keepdim=True) + 1e-10)
    # *****************
    '''
    PLGE_inter_loss_L2 = Source_InterClass_sim_loss(feature_source, label_source, target_Tmean,'adj')
    '''
    norm_h = h_s.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
    norm_s = target_Tmean.pow(2).data.sum(1).pow(1 / 2).unsqueeze(1)
    h_s = h_s.mul(1 /(norm_h + 1e-8))  # 对特征矩阵的数据进行了处理
    target_Tmean = target_Tmean.mul(1 / (norm_s + 1e-8))  # 对源域类中心数据进行了处理
    Flag = torch.ones(h_s.shape).cuda()
    Target_Discri = Flag*target_Tmean
    # a=h_s.mm(Flag.t())
    sim = torch.sum(h_s.mm(Flag.t()), 1) / (h_s.shape[0])

    return sim1,sim


# =================================================================


def weights_init(m):   # 初始化权值，是用来初始化嵌入流行层的
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.1)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0.1)


# ==========================================================
#       Grassmannian Manifold Metric
# ==========================================================


def grassmann_dist_Fast(input1, input2):  # 这里的输入是源域和目标域的协方差矩阵

    fea_dim = input1.shape[1]
    h_src = input1 - torch.mean(input1, dim=0)
    h_trg = input2 - torch.mean(input2, dim=0)

    _, D1, V1 = torch.svd(h_src)  # 协方差矩阵的SVD分解
    _, D2, V2 = torch.svd(h_trg)

    grassmann_dist = torch.sum(torch.pow(V1.mm(V1.t())-V2.mm(V2.t()), 2))/(fea_dim*fea_dim)
    return grassmann_dist, D1, D2

#==========================================================

# ==========================================================
#       Compute Accuracy
# ==========================================================


def classification_accuracy(e, c, data_loader):
    with torch.no_grad():
        correct = 0
        for batch_idx, (X, target) in enumerate(data_loader):
            X, target = Variable(X), Variable(target).long().squeeze()
            X, target = X.cuda(), target.cuda()
            t_fe = e.forward(X)

            output, _, _ = c(t_fe)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        return correct.item() / len(data_loader.dataset)


