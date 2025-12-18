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
import torch.nn as nn
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
manifold_dim = [2048, 1024]
class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor_res50 = model_dict[args.model.base_model](args.model.pretrained_model)
        self.feature_extractor_clip, preprocess = clip.load("/root/BCD_PDA/RN50.pt")  # 加载预训练的CLIP模型
        for param in self.feature_extractor_clip.parameters():
            param.requires_grad = False

        classifier_output_dim = len(source_classes)
        self.clip_classifier = CLS(in_dim=1024, out_dim=classifier_output_dim, bottle_neck_dim=256)
        self.classifier = CLS(in_dim=1024, out_dim=classifier_output_dim,
                              bottle_neck_dim=256)
        # 使用CLIP模型作为辅助分类器
        # self.clip_model, _ = clip.load("/root/BCD_PDA/CLIP_pt/RN50.pt")  # 加载预训练的CLIP模型
        # # self.clip_model.visual.fc = nn.Linear(self.clip_model.visual.fc.in_features, classifier_output_dim) # 确保输出相同
        # self.clip_model.visual.attnpool.c_proj = nn.Linear(in_features=2048, out_features=classifier_output_dim,
        #                                                    bias=True)
        # self.classifier_auxiliary = self.clip_model.visual  # 只使用CLIP的视觉部分
        self.classifier_auxiliary = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, classifier_output_dim),
            TorchLeakySoftmax(classifier_output_dim)
        )

        self.discriminator = AdversarialNetwork(256)
    def forward(self, x):
        with torch.no_grad():
            f = self.feature_extractor_clip.module.encode_image(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        y_aug = self.classifier_auxiliary(x)  # 使用CLIP的视觉模型进行前向传播
        d_aug = self.discriminator(y_aug)  # 使用从CLIP获得的features计算d_aug
        return y, d, y_aug, d_aug

# 后续的代码结构和优化器定义可以保持不变。


totalNet = TotalNet()
# clip_model, preprocess = clip.load("/root/BCD_PDA/CLIP_pt/RN50.pt")  # 加载预训练的CLIP模型

# logger.add_graph(totalNet, torch.ones(2, 3, 224, 224))
feature_extractor_clip = nn.DataParallel(totalNet.feature_extractor_clip, device_ids=[0], output_device=output_device).train(False)
feature_extractor_res50 = nn.DataParallel(totalNet.feature_extractor_res50, device_ids=[0], output_device=output_device).train(False)
# feature_extractor_clip = totalNet.feature_extractor_clip`
classifier = nn.DataParallel(totalNet.classifier, device_ids=[0], output_device=output_device).train(True)
clip_classifier = nn.DataParallel(totalNet.clip_classifier, device_ids=[0], output_device=output_device).train(True)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=[0], output_device=output_device).train(True)
classifier_auxiliary = nn.DataParallel(totalNet.classifier_auxiliary, device_ids=[0], output_device=output_device).train(True)

# Manifold = nn.DataParallel(totalNet.Manifold, device_ids=[0], output_device=output_device).train(True)
if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor_clip.load_state_dict(data['feature_extractor_clip'])
    classifier.load_state_dict(data['classifier'])
    discriminator.load_state_dict(data['discriminator'])
    classifier_auxiliary.load_state_dict(data['classifier_auxiliary'])

    counter = AccuracyCounter()
    with TrainingModeManager([feature_extractor_clip, classifier], train=False) as mgr, torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

            feature = feature_extractor_clip.forward(im)
            ___, __, before_softmax, predict_prob = classifier.forward(feature)

            counter.addOneBatch(variable_to_numpy(predict_prob),
                                variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

    acc_test = counter.reportAccuracy()
    print(f'test accuracy is {acc_test}')
    exit(0)


# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
# optimizer_finetune = OptimWithSheduler(
#     optim.SGD(feature_extractor_clip.parameters(), lr=args.train.lr/10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
#     scheduler)
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
epoch_id = 0
txt_source_label = args.train.txt_source_label
txt_target_label = txt_source_label[:args.data.dataset.n_share]
while global_step < args.train.min_step:

    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1
    source_class_mean_update = (1e-8)*torch.ones(len(source_classes), 256).cuda()   # 第二层流形
    target_Tmean_update = (1e-8) * torch.ones(1, 256).cuda()

    C_num_count = (1e-8) * torch.ones(len(source_classes), 1).cuda()
    # H_mean_update_L1 = torch.zeros(len(source_classes), manifold_dim[0]).cuda()  # 第一层流形

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage
        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        # 构建源域标签
        label_source_txt = [txt_source_label[i] for i in label_source]
        label_source_descriptions = ["this is a photo of {" + label for label in label_source_txt+"}"]
        # label_source_descriptions = [label for label in label_source_txt]
        source_text_tokens = clip.tokenize(label_source_descriptions).cuda()

        # 构建目标域标签
        label_target_txt = [txt_target_label[i] for i in label_target]
        label_target_descriptions = [label for label in label_target_txt]
        target_text_tokens = clip.tokenize(label_target_descriptions).cuda()
        with torch.no_grad():
            # fc1_s = feature_extractor_res50.forward(im_source)
            # fc1_t = feature_extractor_res50.forward(im_target)


            image_features_s = feature_extractor_clip.module.encode_image(im_source)
            image_features_s /= image_features_s.norm(dim=-1, keepdim=True)
            fc1_s = image_features_s

            image_features_t = feature_extractor_clip.module.encode_image(im_target)
            image_features_t /= image_features_t.norm(dim=-1, keepdim=True)
            fc1_t = image_features_t

        # fc1_s, text_features_s, logits_per_image_s, logits_per_text_s = feature_extractor_clip.forward(im_source, target_text_tokens)
        # fc1_t, text_features_t, logits_per_image_t, logits_per_text_t = feature_extractor_clip.forward(im_target, target_text_tokens)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s.float())
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t.float())

        domain_prob_discriminator_source = discriminator.forward(feature_source)
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

        # ================ Source Discriminative Loss ==============
        if (epoch_id) <= 5:  # PLGE_Inter_step = 1 源域的类间损失
            Source_Discri = torch.zeros(1).squeeze(0).cuda()
        else:  # H_Tmean_use_L2 源域中心
            # Source_Discri = Source_InterClass_sim(feature_source, label_source, target_Tmean)
            Source_Discri = Source_InterClass_sim(feature_source, label_source, source_mean)
            if isinstance(Source_Discri, torch.Tensor):
                log_wei.write(f'Source_Discri is {Source_Discri.t()} \n')
            else:
                log_wei.write(f'Source_Discri is {Source_Discri} \n')

            # log_wei.write(f'Source_Discri is {Source_Discri.t()} \n')
        # ===================== Target discriminate Loss =========================
        if (epoch_id) <= 5:  # PLGE_step = 10  ,  PLGE_lambda_L2, PLGE_lambda_L1 = 1e1, 1e0
            Target_Discri = torch.ones_like(domain_prob_source_aug)
        else:  # 源域类中心H_mean_use_L2 对应公式10
            Target_Discri, Source_Discri_target = Target_IntraClass_sim(feature_target.detach(),feature_source.detach(),256, source_class_means,target_Tmean, label_source)
            if isinstance(Source_Discri_target, torch.Tensor):
                log_wei.write(f'Source_Discri_target is {Source_Discri_target.t()} \n')
            else:
                log_wei.write(f'Source_Discri_target is {Source_Discri_target} \n')

            # log_wei.write(f'Source_Discri_target is {Source_Discri_target.t()} \n')
            log_wei.write(f'Target_Discri is {Target_Discri.t()} \n')
        # ==============================compute loss
        weight = (1.0 - domain_prob_source_aug) # 这个就是w1
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
        weight = weight * Target_Discri # w2 其中Target_Discri是目标域特征和源领域类中心的相似性
        log_wei.write(f'finally_weight is {weight.t()} \n\n')
        weight = weight.detach()
        log_wei.flush()
        # ============================== cross entropy loss, it receives logits as its inputs
        ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source).view(-1, 1)
        ce = torch.mean(ce * weight, dim=0, keepdim=True)

        tmp = weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss = torch.mean(tmp, dim=0, keepdim=True)
        adv_loss += nn.BCELoss()(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))

        ce_aug = nn.BCELoss(reduction='none')(predict_prob_source_aug, one_hot(label_source, args.data.dataset.n_total))
        ce_aug = torch.sum(ce_aug) / label_source.numel()
        adv_loss_aug = nn.BCELoss()(domain_prob_source_aug, torch.ones_like(domain_prob_source_aug))#1
        adv_loss_aug += nn.BCELoss()(domain_prob_target_aug, torch.zeros_like(domain_prob_target_aug))#0

        entropy = EntropyLoss(predict_prob_target)

        # Source_Discri_weight = getattr(args.train, 'Source_Discri', 1.0)  # 如果配置文件中没有定义，则默认为1.0

        with OptimizerManager(
                # [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_classifier_auxiliary]):
                [optimizer_cls, optimizer_discriminator, optimizer_classifier_auxiliary]):
            loss = ce + args.train.adv_loss_tradeoff * adv_loss + args.train.entropy_tradeoff * entropy + \
                   args.train.adv_loss_aug_tradeoff * adv_loss_aug + args.train.ce_aug_tradeoff * ce_aug + args.train.Source_Discri_weight * Source_Discri

            # loss = ce + args.train.adv_loss_tradeoff * adv_loss + args.train.entropy_tradeoff * entropy + \
            #        args.train.adv_loss_aug_tradeoff * adv_loss_aug + args.train.ce_aug_tradeoff * ce_aug
            loss.backward()
################################################3
        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            # print("Source_Discri_weight * Source_Discr=", args.train.Source_Discri_weight * Source_Discri)
            # print("args.train.adv_loss_aug_tradeoff * adv_loss_aug=",args.train.adv_loss_aug_tradeoff * adv_loss_aug)
            # print("args.train.ce_aug_tradeoff * ce_aug=",args.train.ce_aug_tradeoff * ce_aug)
            # print("args.train.adv_loss_tradeoff * adv_loss=",args.train.adv_loss_tradeoff * adv_loss)
            # print("args.train.entropy_tradeoff * entropy=",args.train.entropy_tradeoff * entropy)
            print("ce=",ce)
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
            log_train.write(f'epoch_id:{epoch_id} ||acc_train is {acc_train.item()} \n')
            log_train.flush()
        if global_step % args.test.test_interval == 0:

            counter = AccuracyCounter()
            # with TrainingModeManager([feature_extractor_clip, classifier], train=False) as mgr, torch.no_grad():
            #
            #     for i, (im, label) in enumerate(target_train_dl):
            #         im = im.to(output_device)
            #         label = label.to(output_device)
            #
            #         feature = feature_extractor_clip.forward(im)
            #         ___, __, before_softmax, predict_prob = classifier.forward(feature)
            #
            #         counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))
            with TrainingModeManager([feature_extractor_clip, classifier], train=False) as mgr, torch.no_grad():

                for i, (im, label) in enumerate(target_train_dl):
                    im = im.to(output_device)
                    label = label.to(output_device)

                    label_target_txt = [txt_target_label[i] for i in label]
                    label_target_descriptions = [label for label in label_target_txt]
                    source_target_tokens = clip.tokenize(label_target_descriptions).cuda()

                    # feature, _, _ ,_ = feature_extractor_clip.e(im,source_target_tokens)
                    feature = feature_extractor_clip.module.encode_image(im)
                    feature /= feature.norm(dim=-1, keepdim=True)

                    ___, __, before_softmax, predict_prob = classifier.forward(feature.float())

                    counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

            acc_test = counter.reportAccuracy()

            logger.add_scalar('acc_test', acc_test, global_step)
            log_text.write(f'acc_test is {acc_test} \n')
            log_text.flush()
            clear_output()

            data = {
                "feature_extractor_clip": feature_extractor_clip.state_dict(),
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