from easydl import inverseDecaySheduler, OptimWithSheduler, OptimizerManager, one_hot, TorchLeakySoftmax
from data import *
from net import *
import datetime
from tqdm import tqdm
import network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import torch
import os
import seaborn as sns

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
import matplotlib.lines as mlines
import numpy as np
from matplotlib.image import imread
import numpy as np
# from net import mixup_data
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'

font_path = '/root/Times-New-Roman-Bold.ttf'
font_prop = FontProperties(fname=font_path, weight='bold')
plt.rcParams['font.family'] = font_prop.get_name()


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



def visualize_tsne(feature_extractor, classifier, source_train_dl, target_train_dl, args):
    # 确保模型处于评估模式
    feature_extractor.eval()
    classifier.eval()

    # Initialize lists to store features, labels, and domain information
    source_features = []
    target_features = []
    source_labels = []
    target_labels = []

    # Collect features and labels for the target domain
    for im, label in tqdm(target_train_dl, desc='Target Domain'):
        im = im.to(output_device)
        label = label.to(output_device)
        feature = feature_extractor(im)
        _, feature, _, _ = classifier(feature)
        target_features.append(feature.detach().cpu().numpy())
        target_labels.append(label.cpu().numpy())

    # Collect features and labels for the source domain
    for im, label in tqdm(source_train_dl, desc='Source Domain'):
        im = im.to(output_device)
        label = label.to(output_device)
        feature = feature_extractor(im)
        _, feature, _, _ = classifier(feature)
        source_features.append(feature.detach().cpu().numpy())
        source_labels.append(label.cpu().numpy())

    # Convert lists to numpy arrays
    source_features = np.vstack(source_features)
    target_features = np.vstack(target_features)
    source_labels = np.hstack(source_labels)
    target_labels = np.hstack(target_labels)

    # Combine features from both domains
    combined_features = np.vstack([source_features, target_features])
    combined_labels = np.hstack([source_labels, target_labels])

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
    tsne_results = tsne.fit_transform(combined_features)

    # Create a color palette with seaborn
    palette = np.array(sns.color_palette("hsv", args.data.dataset.n_total))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    # 我们需要在正确的范围内使用布尔索引
    for i in range(10):
        # Source domain
        idxs = (source_labels == i)
        if idxs.any():
            center = np.mean(tsne_results[:len(source_features)][idxs, :], axis=0)
            plt.text(center[0], center[1], f'S{i}', color='black', fontsize=16, ha='center', va='center')

        # Target domain
        idxs = (target_labels == i)
        if idxs.any():
            center = np.mean(tsne_results[len(source_features):][idxs, :], axis=0)
            plt.text(center[0], center[1], f'T{i}', color='black', fontsize=16, ha='center', va='center')

    scatter_source = ax.scatter(tsne_results[:len(source_features), 0], tsne_results[:len(source_features), 1],
                          c=palette[source_labels.astype(int)],
                          alpha=0.2,
                          label="Source",
                          edgecolors='w',
                          linewidth=0.5,s=50)

    scatter_target = ax.scatter(tsne_results[len(source_features):, 0], tsne_results[len(source_features):, 1],
                          c=palette[target_labels.astype(int)],
                          alpha=1,
                          label="Target",
                          marker='+',s=80)

    # # Add legend for classes
    # class_legends = [plt.Line2D([0], [0], marker='o', color='w', label=str(i),
    #                             markerfacecolor=palette[i], markersize=5) for i in range(args.data.dataset.n_total)]
    # plt.legend(handles=class_legends, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=16)
    # 创建图例
    legend_elements = [
        mlines.Line2D([0], [0], color='w', marker='o', markerfacecolor='red', label='Source (S)', markersize=8),
        mlines.Line2D([0], [0], color='w', marker='+', markeredgecolor='red', label='Target (T)', markersize=8),
    ]

    # 在适当位置添加图例
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    plt.title(r'ResNet t-SNE Visualization On Office-31 Dataset', fontsize=18, fontproperties=font_prop)
    # Save the plot with high dpi
    # pdf_path = '/root/autodl-tmp/BCD_PDA/office_31_log/20231108/DCAW/amazon&&webcam/tsne_ETN.pdf'

    pdf_path = '/root/autodl-tmp/BCD_PDA/office_31_log/20231108/Res50/tsne_Res50.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=600)
    # plt.savefig('/root/autodl-tmp/BCD_PDA/office_31_log/20231108/DACW/amazon&&webcam/tsne.png', dpi=1200)
    plt.show()

    # # 在目标领域的每个类别中心添加标签
    # for i in range(10):  # 这里我们只为共有的10个类别添加标签
    #     idxs = source_labels == i
    #     if idxs.any():
    #         center = np.mean(tsne_results[len(source_features):][idxs, :], axis=0)
    #         ax.text(center[0], center[1], f's{i}', color='blue', fontsize=14, ha='center', va='center')
    #
    # # 在目标领域的每个类别中心添加标签
    # for i in range(10):  # 这里我们只为共有的10个类别添加标签
    #     idxs = target_labels == i
    #     if idxs.any():
    #         center = np.mean(tsne_results[len(source_features):][idxs, :], axis=0)
    #         ax.text(center[0], center[1], f't{i}', color='red', fontsize=14, ha='center', va='center')
    # 为源领域和目标领域添加图例
    # legend1 = ax.legend(*scatter_source.legend_elements(), loc="upper left", title="Source Classes")
    # legend2 = ax.legend(*scatter_target.legend_elements(), loc="upper right", title="Target Classes")
    # ax.add_artist(legend1)
    # ax.add_artist(legend2)


def main():
    # 参数设置

    # 模型初始化
    totalNet = TotalNet()

    feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=[0], output_device=output_device).train(
        True)
    classifier = nn.DataParallel(totalNet.classifier, device_ids=[0], output_device=output_device).train(True)

    data = torch.load(open(args.test.resume_file, 'rb'))
    # 如果是res50就注释掉
    # feature_extractor.load_state_dict(data['feature_extractor'])
    # classifier.load_state_dict(data['classifier'])


    # 执行评估和可视化
    visualize_tsne(feature_extractor, classifier, source_train_dl, target_train_dl, args)

if __name__ == "__main__":
    main()
