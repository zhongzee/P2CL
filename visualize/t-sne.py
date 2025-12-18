from easydl import inverseDecaySheduler, OptimWithSheduler, OptimizerManager, one_hot, TorchLeakySoftmax
from data import *
from net import *
import datetime
from tqdm import tqdm
import network
if is_in_notebook():

    from tqdm import tqdm_notebook as tqdm
from torch import optim, nn
import torch
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True
import net
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.image import imread
import numpy as np
# from net import mixup_data
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import cm

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)   # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)  # 如果读取数据的过程采用了随机预处理(如RandomCrop、RandomHorizontalFlip等)，那么对python、numpy的随机数生成器也需要设置种子。
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)   # 为了禁止hash随机化，使得实验可复现。
import Utilizes
seed_everything()
# import torchsnooper
output_device = torch.device("cuda:0")

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        self.classifier_auxiliary = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, classifier_output_dim),
            TorchLeakySoftmax(classifier_output_dim)
        )
        # self.Manifold = Manifold(256,manifold_dim[0],manifold_dim[1],classifier_output_dim)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        y_aug, d_aug = self.classifier_auxiliary(_)
        # w2 = self.Manifold(_)
        return y, d, y_aug, d_aug

totalNet = TotalNet()

feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=[0], output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=[0], output_device=output_device).train(True)

data = torch.load(open(args.test.resume_file, 'rb'))
# 如果是res50就注释掉
feature_extractor.load_state_dict(data['feature_extractor'])
classifier.load_state_dict(data['classifier'])

counter = AccuracyCounter()
with torch.no_grad():

    for i, (im, label) in enumerate(tqdm(target_train_dl, desc='testing ')):
        im = im.to(output_device)
        label = label.to(output_device)
        feature = feature_extractor.forward(im)
        _, feature, before_softmax, predict_prob = classifier.forward(feature)

        counter.addOneBatch(variable_to_numpy(predict_prob),
                            variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

        if i == 0:
            X = np.array(feature.cpu().numpy())
            label2 = tuple(label)
        else:
            feature = np.array(feature.data.cpu().numpy())
            X = np.row_stack((X, feature))
            label = tuple(label)
            label2 = label2 + label
    j = 0

    for i, (im, label) in enumerate(tqdm(source_train_dl, desc='training ')):
        im = im.to(output_device)
        label = label.to(output_device)
        feature = feature_extractor.forward(im)
        _, feature, before_softmax, predict_prob = classifier.forward(feature)  # note！
        if i == 0:
            X1 = np.array(feature.data.cpu().numpy())
            label1 = tuple(label)
        else:
            feature = np.array(feature.data.cpu().numpy())
            label = tuple(label)
            X1 = np.row_stack((X1, feature))
            label1 = label1+label

    # label1 = label1.cpu().numpy()
    # model = TSNE(perplexity=30.0)
    # np.set_printoptions(suppress=True)
    # X = torch.from_numpy(X)
    # dim1= X.shape[0]
    # X1 = torch.from_numpy(X1)
    # dim2=X1.shape[0]
    # X1 = torch.cat((X,X1))
    # label = label2 + label1
    # Y = model.fit_transform(X1)  # 将X降维(默认二维)后保存到Y中
    #
    # j = 0
    # plt.scatter(Y[dim1:dim2, 0], Y[dim1:dim2, 1], s=5, c="b")  # w-d
    # plt.scatter(Y[0:dim1, 0], Y[0:dim1, 1], s=5, c="red")
    # plt.legend(['source', 'target'])  # 添加图例
    # plt.title('T-SNE')

    #################
    # 应用 t-SNE
    model = TSNE(perplexity=30.0, random_state=42)
    X_combined = np.vstack([X, X1])
    Y = model.fit_transform(X_combined)

    # 源领域和目标领域数据点的数量
    dim1 = X.shape[0]
    dim2 = X1.shape[0]
    # 选择 colormap
    color_map = cm.get_cmap('tab10')

    # 设置图表的背景颜色和网格
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 绘制 t-SNE 散点图，源领域使用方框（marker='s'），目标领域使用三角形（marker='^'）
    plt.scatter(Y[:dim1, 0], Y[:dim1, 1], s=50, c=color_map(0.1), marker='s', edgecolor='black', label='Target')
    plt.scatter(Y[dim1:dim1 + dim2, 0], Y[dim1:dim1 + dim2, 1], s=50, c=color_map(0.2), marker='^', edgecolor='black',
                label='Source')

    # 添加图例和标题
    plt.legend(loc='best')
    plt.title('t-SNE Visualization')

    # 显示图表
    plt.show()

    #################
    import os
    log_dir = os.path.join(args.log.root_dir, f'{source_domain_name}&&{target_domain_name}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 创建目录

    name = f'{source_domain_name}&&{target_domain_name}_current_tsne.JPEG'
    file_path = os.path.join(log_dir, name)  # 使用os.path.join来确保路径正确合并
    plt.savefig(file_path, dpi=600, format='JPEG')  # 保存图像



