import torch

from data import *
from net import *
import datetime
from tqdm import tqdm
# if is_in_notebook():
#     from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True
from easydl import inverseDecaySheduler, OptimWithSheduler, OptimizerManager, one_hot, TorchLeakySoftmax
import clip
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count
import json
import os


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
output_device = torch.device("cuda:0")

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything()

now = datetime.datetime.now().strftime('%b%d&%H&%M&%S')

log_dir = f'{args.log.root_dir}/{source_domain_name}&&{target_domain_name}/{now}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

log_text = open(join(log_dir, 'log.txt'), 'w')
log_wei = open(join(log_dir, 'log_wei.txt'), 'w')
log_train = open(join(log_dir, 'log_train.txt'), 'w')
model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc,
    'alexnet':AlexNetFc
}
manifold_dim = [2048,1024]

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        if args.use_clip:
            self.feature_extractor_clip, preprocess = clip.load("/root/BCD_PDA/RN50.pt")  # 加载预训练的CLIP模型 ViT-B-32
            for param in self.feature_extractor_clip.parameters():
                param.requires_grad = False
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.classifier_clip = CLS(1024, classifier_output_dim, bottle_neck_dim=256)
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
        ) # 把这个换成CLIP
        # self.Manifold = Manifold(256,manifold_dim[0],manifold_dim[1],classifier_output_dim)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)  # y分类
        d = self.discriminator(_)  # 判别器 0或1
        y_aug, d_aug = self.classifier_auxiliary(_)  # 辅助分类
        if args.use_clip:
            with torch.no_grad():
                f_clip = self.feature_extractor_clip.encode_image(x)
            f_clip, _clip, __clip, y_clip = self.classifier(f_clip)  # y分类
            d_clip = self.discriminator(_clip)  # 判别器 0或1
            y_aug_clip, d_aug_clip = self.classifier_auxiliary(_)  # 辅助分类
        # w2 = self.Manifold(_)
            return [y, d, y_aug, d_aug], [y_clip, d_clip, y_aug_clip, d_aug_clip]#分类结果，域判别结果，辅助分类结果，辅助域判别结果
        else:
            return [y, d, y_aug, d_aug]


from fvcore.nn import FlopCountAnalysis, parameter_count

def calculate_model_complexity(model, input_size=(1, 3, 224, 224)):
    # Create a dummy input based on the input size
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)  # Ensure dummy_input is on the same device as the model

    # Calculate the FLOPs for the given model and input
    flops = FlopCountAnalysis(model, dummy_input).total()

    # Calculate the total number of parameters, including non-trainable ones
    params = sum(p.numel() for p in model.parameters())

    # Create a dictionary to hold the complexity metrics
    complexity = {
        "params": params,  # Total number of parameters
        "flops": flops
    }

    return complexity

# Now call the function and print the complexity
totalNet = TotalNet()  # Ensure TotalNet is properly defined and instantiated
complexity = calculate_model_complexity(totalNet)
print("model 初始参数和flops:",complexity)  # Should print the correct number of parameters and FLOPs

def aggregate_and_save_class_weights(epoch_id, global_step, label_source, weight, class_weights, class_counts, save_step=500, num_classes=31, file_path='class_weights.json'):
    """
    Aggregate weights for each class and save them at specified intervals.

    :param epoch_id: Current epoch number.
    :param global_step: Current global step in the training loop.
    :param label_source: The labels of the source domain data.
    :param weight: The current weight vector for the batch.
    :param class_weights: Accumulated weights for each class.
    :param class_counts: Counts of occurrences for each class.
    :param save_step: Number of steps between saves.
    :param num_classes: Total number of classes.
    :param file_path: File path to save the JSON data.
    """

    # Aggregate weights based on the source domain labels
    for j in range(weight.size(0)):  # Iterate over batch samples
        class_idx = label_source[j].item()  # Get class index for each sample
        class_weights[class_idx] += weight[j].cpu()  # Accumulate weight
        class_counts[class_idx] += 1  # Increment count

    # Save the weights at specified intervals
    if global_step % save_step == 0:
        # Calculate accumulated weight per class
        accumulated_weights = class_weights / class_counts
        # Convert to list for JSON serialization
        weights_list = accumulated_weights.squeeze().tolist()  # Remove dimensions of size 1 and convert to list

        # Load existing data if file exists, otherwise initialize an empty dict
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # Save the current weights under the iteration number
        data[str(global_step)] = weights_list

        # Write the updated data back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f)

        print(f"Saved class weights to {file_path} at global step {global_step}")

    return class_weights, class_counts


# logger.add_graph(totalNet, torch.ones(2, 3, 224, 224))
feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=[0], output_device=output_device).train(True)
if args.use_clip:
    feature_extractor_clip = totalNet.feature_extractor_clip.to(output_device)
classifier = nn.DataParallel(totalNet.classifier, device_ids=[0], output_device=output_device).train(True)
classifier_clip = nn.DataParallel(totalNet.classifier_clip, device_ids=[0], output_device=output_device).train(True)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=[0], output_device=output_device).train(True)
classifier_auxiliary = nn.DataParallel(totalNet.classifier_auxiliary, device_ids=[0], output_device=output_device).train(True)

# Manifold = nn.DataParallel(totalNet.Manifold, device_ids=[0], output_device=output_device).train(True)
if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    discriminator.load_state_dict(data['discriminator'])
    classifier_auxiliary.load_state_dict(data['classifier_auxiliary'])

    counter = AccuracyCounter()
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

            feature = feature_extractor.forward(im)
            ___, __, before_softmax, predict_prob = classifier.forward(feature)

            counter.addOneBatch(variable_to_numpy(predict_prob),
                                variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

    acc_test = counter.reportAccuracy()
    print(f'test accuracy is {acc_test}')
    exit(0)

# if args.resume.is_resume:
#     assert os.path.exists(args.test.resume_file)
#     data = torch.load(open(args.test.resume_file, 'rb'))
#     feature_extractor.load_state_dict(data['feature_extractor'])
#     classifier.load_state_dict(data['classifier'])
#     discriminator.load_state_dict(data['discriminator'])
#     classifier_auxiliary.load_state_dict(data['classifier_auxiliary'])

# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr/10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.train.lr , weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_classifier_auxiliary = OptimWithSheduler(
    optim.SGD(classifier_auxiliary.parameters(), lr=args.train.lr/10.0 , weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
# optimizer_Manifold = OptimWithSheduler(
#     optim.SGD(Manifold.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
#     scheduler)
global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 4 # 0
if args.use_clip:
    txt_source_label = args.train.txt_source_label
    txt_target_label = txt_source_label[:args.data.dataset.n_share]

# Define the function to compute cosine similarity (this will be part of utility functions in the script)
# Define the function to compute cosine similarity (this will be part of utility functions in the script)
def cosine_similarity(x, y):
    x = x.float()
    y = y.float()
    # Ensure the feature vectors are L2-normalized
    x = x / x.norm(dim=1, keepdim=True)
    y = y / y.norm(dim=1, keepdim=True)
    return x @ y.T

if args.use_clip:
    templates_source = args.train.templates[str(args.data.dataset.source)]
    templates_target = args.train.templates[str(args.data.dataset.target)]

while global_step < args.train.min_step:

    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1
    source_class_mean_update = (1e-8)*torch.ones(len(source_classes), 256).cuda()   # 第二层流形
    target_Tmean_update = (1e-8) * torch.ones(1, 256).cuda()

    C_num_count = (1e-8) * torch.ones(len(source_classes), 1).cuda()
    # H_mean_update_L1 = torch.zeros(len(source_classes), manifold_dim[0]).cuda()  # 第一层流形

    # 在训练循环开始前初始化
    num_classes = args.data.dataset.n_total  # 根据您的实际类别数更改
    class_weights = torch.zeros(num_classes, 1)
    class_counts = torch.zeros(num_classes, 1)

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        fc1_s = feature_extractor.forward(im_source)#Out[1]: torch.Size([36, 2048])
        fc1_t = feature_extractor.forward(im_target)
        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)#Out[6]: torch.Size([36, 2048])
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_discriminator_source = discriminator.forward(feature_source) # #Out[1]: torch.Size([36, 1])
        domain_prob_discriminator_target = discriminator.forward(feature_target)

        predict_prob_source_aug, domain_prob_source_aug = classifier_auxiliary.forward(feature_source.detach())
        predict_prob_target_aug, domain_prob_target_aug = classifier_auxiliary.forward(feature_target.detach())

        # # =============== compute the class mean vector =========
        sam_count = 0
        Tensor_size = label_source.shape  # 源域标签
        if Tensor_size:
            target = label_source
        else:
            target = label_source.unsqueeze(0)

        for i in target:
            C_num_count[i] += 1
            source_class_mean_update[i, :] += feature_source[sam_count, :].data  # 改用feature_source
            target_Tmean_update += feature_target[sam_count,:].data
            sam_count += 1
        # print('C_num_count:',C_num_count)
        # ===================== Align Loss =========================
        if (epoch_id) <= 5:  # Aligned_step = 0
            Coral_Grass_loss_L1 = torch.zeros(1).squeeze(0).cuda()
            # Coral_Grass_loss_L2 = torch.zeros(1).squeeze(0).cuda()
        else:  # 直接跑这里
            Coral_Grass_loss_L1, _, _ = grassmann_dist_Fast(feature_source, feature_target)  # 每一层流行层的距离度量
            # Coral_Grass_loss_L2 = grassmann_dist_Fast(H_mean_use_L2, feature_target, label_source,256)
        Align_loss = Coral_Grass_loss_L1
        # Align features by maximizing the similarity with corresponding text prompts
        # This can be part of the loss function during training

        # ================ Source Discriminative Loss ==============
        if (epoch_id) <= 5:  # PLGE_Inter_step = 1 源域的类间损失
            Source_Discri = torch.zeros(1).squeeze(0).cuda()
        else:  # H_Tmean_use_L2 源域中心
            # Source_Discri = Source_InterClass_sim(feature_source, label_source, target_Tmean)
            Source_Discri = Source_InterClass_sim(feature_source, label_source, source_mean)
            log_wei.write(f'Source_Discri is {Source_Discri.t()} \n')
        # ===================== Target discriminate Loss =========================
        if (epoch_id) <= 5:  # PLGE_step = 10  ,  PLGE_lambda_L2, PLGE_lambda_L1 = 1e1, 1e0
            Target_Discri = torch.ones_like(domain_prob_source_aug)
        else:  # 源域类中心H_mean_use_L2
            Target_Discri, Source_Discri_target = Target_IntraClass_sim(feature_target.detach(),feature_source.detach(),256, source_class_means,target_Tmean, label_source)
            log_wei.write(f'Source_Discri_target is {Source_Discri_target.t()} \n')
            log_wei.write(f'Target_Discri is {Target_Discri.t()} \n')
        # ==============================compute loss
        weight = (1.0 - domain_prob_source_aug)
        # weight = weight
        log_wei.write(f'*************************************\n')
        log_wei.write(f'*************************************\n')
        log_wei.write(f'domain_prob_source_aug is {domain_prob_source_aug} \n')
        # print('w1:',weight.t())
        # print('Target_Discri:',Target_Discri.t())
        # print('Align_loss:',Align_loss*1000)
        weight = weight / (torch.mean(weight, dim=0, keepdim=True) + 1e-8)
        log_wei.write(f'epoch is {epoch_id} \n')
        log_wei.write(f'Align_loss is {Align_loss} \n')
        log_wei.write(f's_label is {label_source} \n')
        log_wei.write(f'weight is {weight} \n')
        weight = weight * Target_Discri
        log_wei.write(f'finally_weight is {weight.t()} \n\n')
        weight = weight.detach()
        log_wei.flush()

        # 调用函数来聚合和保存权重
        class_weights, class_counts = aggregate_and_save_class_weights(epoch_id, global_step, label_source,
                                                                       weight, class_weights, class_counts,
                                                                       save_step=500, num_classes=31,
                                                                       file_path=log_dir + '/class_weights.json')
        print(log_dir+"/class_weights.json")
        # if args.use_clip:
        #     # 计算加权特征对齐损失
        #     feature_alignment_loss = (feature_alignment_loss_source * weight).mean() + \
        #                              (feature_alignment_loss_target * weight).mean()
        #     log_wei.write(f'feature_alignment_loss is {feature_alignment_loss} \n')

        # ============================== cross entropy loss, it receives logits as its inputs
        # ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source).view(-1, 1)
        ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source).view(-1, 1)
        ce = torch.mean(ce * weight, dim=0, keepdim=True) #分类损失 公式3

        tmp = weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss = torch.mean(tmp, dim=0, keepdim=True)  #对抗损失 公式4
        adv_loss += nn.BCELoss()(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))

        ce_aug = nn.BCELoss(reduction='none')(predict_prob_source_aug, one_hot(label_source, args.data.dataset.n_total))
        ce_aug = torch.sum(ce_aug) / label_source.numel()
        adv_loss_aug = nn.BCELoss()(domain_prob_source_aug, torch.ones_like(domain_prob_source_aug))#1
        adv_loss_aug += nn.BCELoss()(domain_prob_target_aug, torch.zeros_like(domain_prob_target_aug))#0

        entropy = EntropyLoss(predict_prob_target)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_classifier_auxiliary]):
            loss = ce + args.train.adv_loss_tradeoff * adv_loss + args.train.entropy_tradeoff * entropy + \
                   args.train.adv_loss_aug_tradeoff * adv_loss_aug + args.train.ce_aug_tradeoff * ce_aug + args.train.Source_Discri_weight * Source_Discri

            loss.backward()

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            # print("Source_Discri_weight * Source_Discr=", args.train.Source_Discri_weight * Source_Discri)
            # print("args.train.adv_loss_aug_tradeoff * adv_loss_aug=",args.train.adv_loss_aug_tradeoff * adv_loss_aug)
            # print("args.train.ce_aug_tradeoff * ce_aug=",args.train.ce_aug_tradeoff * ce_aug)
            # print("args.train.adv_loss_tradeoff * adv_loss=",args.train.adv_loss_tradeoff * adv_loss)
            # print("args.train.entropy_tradeoff * entropy=",args.train.entropy_tradeoff * entropy)
            print("ce=",ce)
            print("adv_loss=", adv_loss)
            print("entropy=", entropy)
            print("adv_loss_aug=", adv_loss_aug)
            print("ce_aug=", ce_aug)
            print("Source_Discri=", Source_Discri)

            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
            print('acc_train:',acc_train.item())
            logger.add_scalar('entropy', entropy, global_step)
            logger.add_scalar('adv_loss', adv_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            logger.add_scalar('adv_loss_aug', adv_loss_aug, global_step)
            logger.add_scalar('ce_aug', ce_aug, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)
            logger.add_scalar('Source_Discri', Source_Discri, global_step)
            # if args.use_clip:
            #     logger.add_scalar('feature_alignment_source', (feature_alignment_loss_source * weight).mean(), global_step)
            #     logger.add_scalar('feature_alignment_target', (feature_alignment_loss_target * weight).mean(), global_step)
            #     logger.add_scalar('feature_alignment_loss', feature_alignment_loss, global_step)
            # logger.add_scalar('grassmann_Align_loss', Align_loss, global_step)
            log_train.write(f'epoch_id:{epoch_id} ||acc_train is {acc_train.item()} \n')
            log_train.flush()
        if global_step % args.test.test_interval == 0:

            counter = AccuracyCounter()
            with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():

                for i, (im, label) in enumerate(target_train_dl):
                    im = im.to(output_device)
                    label = label.to(output_device)

                    feature = feature_extractor.forward(im)
                    ___, __, before_softmax, predict_prob = classifier.forward(feature)
                    # feature = feature_extractor_clip.encode_image(im)
                    # ___, __, before_softmax, predict_prob = classifier_clip.forward(feature.float())

                    counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

            acc_test = counter.reportAccuracy()

            logger.add_scalar('acc_test', acc_test, global_step)
            log_text.write(f'acc_test is {acc_test} \n')
            log_text.flush()
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator.state_dict(),
                'classifier_auxiliary': classifier_auxiliary.state_dict(),
            }

            if acc_test > best_acc:
                best_acc = acc_test

                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)
            print("\n current_test:",acc_test)
            print("best_test:",best_acc)
            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)

    source_class_means = source_class_mean_update.mul(1 / C_num_count)  # Class mean matrix
    source_mean = source_class_mean_update.mean(0)  # Total mean vector 域均值?
    target_Tmean = target_Tmean_update / len(target_train_ds)
    # H_mean_use_L1 = H_mean_update_L1.mul(1 / C_num_count)
    # H_Tmean_use_L1 = H_mean_update_L1.mean(0)
    # del source_mean