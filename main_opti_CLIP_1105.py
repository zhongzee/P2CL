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
import optuna
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
output_device = torch.device("cuda:0")
import clip
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count
import json

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

# now = datetime.datetime.now().strftime('%b%d&%H&%M&%S')
#
# log_dir = f'{args.log.root_dir}/{source_domain_name}&&{target_domain_name}/{now}'
#
# logger = SummaryWriter(log_dir)
#
# with open(join(log_dir, 'config.yaml'), 'w') as f:
#     f.write(yaml.dump(save_config))

# log_text = open(join(log_dir, 'log.txt'), 'w')
# log_wei = open(join(log_dir, 'log_wei.txt'), 'w')
# log_train = open(join(log_dir, 'log_train.txt'), 'w')
model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc,
    'alexnet':AlexNetFc
}
# manifold_dim = [2048,1024]

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        if args.use_clip:
            self.feature_extractor_clip, preprocess = clip.load("/root/BCD_PDA/RN50.pt")  # 加载预训练的CLIP模型
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
        f, _, __, y = self.classifier(f)# y分类
        d = self.discriminator(_) # 判别器 0或1
        y_aug, d_aug = self.classifier_auxiliary(_) # 辅助分类
        # w2 = self.Manifold(_)
        return y, d, y_aug, d_aug#分类结果，域判别结果，辅助分类结果，辅助域判别结果

totalNet = TotalNet()
# logger.add_graph(totalNet, torch.ones(2, 3, 224, 224))
feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=[0], output_device=output_device).train(True)
if args.use_clip:
    feature_extractor_clip = totalNet.feature_extractor_clip.to(output_device)
classifier = nn.DataParallel(totalNet.classifier, device_ids=[0], output_device=output_device).train(True)
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

if args.use_clip:
    txt_source_label = args.train.txt_source_label
    txt_target_label = txt_source_label[:args.data.dataset.n_share]
def cosine_similarity(x, y):
    x = x.float()
    y = y.float()
    # Ensure the feature vectors are L2-normalized
    x = x / x.norm(dim=1, keepdim=True)
    y = y / y.norm(dim=1, keepdim=True)
    return x @ y.T
def train_model(args, source_train_dl, target_train_dl, feature_extractor, classifier, discriminator, classifier_auxiliary, log_dir, output_device):

    templates_source = args.train.templates[str(args.data.dataset.source)]
    templates_target = args.train.templates[str(args.data.dataset.target)]

    logger = SummaryWriter(log_dir)
    log_text = open(join(log_dir, 'log.txt'), 'w')
    log_wei = open(join(log_dir, 'log_wei.txt'), 'w')
    log_train = open(join(log_dir, 'log_train.txt'), 'w')
    model_dict = {
        'resnet50': ResNet50Fc,
        'vgg16': VGG16Fc,
        'alexnet': AlexNetFc
    }
    manifold_dim = [2048, 1024]

    seed_everything()

    total_steps = tqdm(range(args.train.min_step), desc='global step')
    global_step = 0
    best_acc = 0
    epoch_id = 0
    source_class_means = None
    source_mean = None
    target_Tmean = None

    while global_step < args.train.min_step:

        iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ',
                     total=min(len(source_train_dl), len(target_train_dl)))
        epoch_id += 1
        source_class_mean_update = (1e-8) * torch.ones(len(source_classes), 256).cuda()  # 第二层流形
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
            if args.use_clip:
                # Assuming 'label_source_txt' contains the class labels from the source domain ###############################CLIP######################################3
                label_source_txt = [txt_source_label[i] for i in label_source]  # Replace with actual source labels
                label_source_descriptions = []
                label_target_descriptions = []
                for label in label_source_txt:
                    label_source_descriptions.extend([template.format(label) for template in templates_source])
                    label_target_descriptions.extend([template.format(label) for template in templates_target])

                # Tokenize and encode text descriptions with CLIP
                # Add .cuda() if running on GPU
                source_text_tokens = clip.tokenize(label_source_descriptions).cuda()
                source_text_embeddings = feature_extractor_clip.encode_text(source_text_tokens)

                target_text_tokens = clip.tokenize(label_target_descriptions).cuda()
                target_text_embeddings = feature_extractor_clip.encode_text(target_text_tokens)

                with torch.no_grad():
                    fc1_s_clip = feature_extractor_clip.encode_image(im_source).float()
                    fc1_t_clip = feature_extractor_clip.encode_image(im_target).float()

                # 计算源域特征与源域文本提示之间的相似度
                cos_sim_source = cosine_similarity(fc1_s_clip.float(), source_text_embeddings)
                # 将相似度转换为权重
                # source_weights = F.softmax(cos_sim_source, dim=-1)
                # # 计算加权特征对齐损失
                # feature_alignment_loss_source = -(cos_sim_source * source_weights).sum(dim=-1).mean()

                feature_alignment_loss_source = cos_sim_source
                # 计算目标域特征与源域文本提示之间的相似度
                cos_sim_target = cosine_similarity(fc1_t_clip.float(), target_text_embeddings)
                # 将相似度转换为权重
                # target_weights = F.softmax(cos_sim_target, dim=-1)
                # # 计算加权特征对齐损失
                # feature_alignment_loss_target = -(cos_sim_target * target_weights).sum(dim=-1).mean()

                feature_alignment_loss_target = cos_sim_target
            #############################################################################

            fc1_s = feature_extractor.forward(im_source)  # Out[1]: torch.Size([36, 2048])
            fc1_t = feature_extractor.forward(im_target)

            fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(
                fc1_s)  # Out[6]: torch.Size([36, 2048])
            fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

            domain_prob_discriminator_source = discriminator.forward(feature_source)  # #Out[1]: torch.Size([36, 1])
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
                target_Tmean_update += feature_target[sam_count, :].data
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
                if isinstance(Source_Discri, torch.Tensor):
                    log_wei.write(f'Source_Discri is {Source_Discri.t()} \n')
                else:
                    log_wei.write(f'Source_Discri is {Source_Discri} \n')

                # log_wei.write(f'Source_Discri is {Source_Discri.t()} \n')
            # ===================== Target discriminate Loss =========================
            if (epoch_id) <= 5:  # PLGE_step = 10  ,  PLGE_lambda_L2, PLGE_lambda_L1 = 1e1, 1e0
                Target_Discri = torch.ones_like(domain_prob_source_aug)
            else:  # 源域类中心H_mean_use_L2 # Target_Discri是目标领域特征和源领域类中心之间的相似度，Source_Discri_target是源领域与目标领域类中心相似度，只使用了Target_Discri
                Target_Discri, Source_Discri_target = Target_IntraClass_sim(feature_target.detach(),
                                                                            feature_source.detach(), 256,
                                                                            source_class_means, target_Tmean,
                                                                            label_source)
                if isinstance(Source_Discri_target, torch.Tensor):
                    log_wei.write(f'Source_Discri_target is {Source_Discri_target.t()} \n')
                else:
                    log_wei.write(f'Source_Discri_target is {Source_Discri_target} \n')

                # log_wei.write(f'Source_Discri_target is {Source_Discri_target.t()} \n')
                log_wei.write(f'Target_Discri is {Target_Discri.t()} \n')
            # ==============================compute loss
            weight = (1.0 - domain_prob_source_aug)  # 这个就是w1 domain_prob_source_aug是D1的输出，给共享类加权
            # weight = weight
            log_wei.write(f'*************************************\n')
            log_wei.write(f'*************************************\n')
            log_wei.write(f'domain_prob_source_aug is {domain_prob_source_aug} \n')
            # print('w1:',weight.t())
            # print('Target_Discri:',Target_Discri.t())
            # print('Align_loss:',Align_loss*1000)
            weight = weight / (torch.mean(weight, dim=0, keepdim=True) + 1e-8)  #
            log_wei.write(f'epoch is {epoch_id} \n')
            log_wei.write(f'Align_loss is {Align_loss} \n')
            log_wei.write(f's_label is {label_source} \n')
            log_wei.write(f'weight is {weight} \n')
            weight = weight * Target_Discri  # Target_Discri是找到目标领域哪些与源领域的哪些类别更相似（共享类），这相当于也是在给目标领域的共享类加权
            # 计算加权特征对齐损失
            if args.use_clip:
                feature_alignment_loss = (feature_alignment_loss_source * weight).mean() + \
                                         (feature_alignment_loss_target * weight).mean()

            #
            # feature_alignment_loss = (feature_alignment_loss_source).mean() + \
            #                          (feature_alignment_loss_target).mean()

            log_wei.write(f'finally_weight is {weight.t()} \n\n')
            weight = weight.detach()
            # log_wei.write(f'feature_alignment_loss is {feature_alignment_loss} \n')
            log_wei.flush()
            # ============================== cross entropy loss, it receives logits as its inputs
            # ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source).view(-1, 1)
            ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source).view(-1, 1)
            ce = torch.mean(ce * weight, dim=0, keepdim=True)  # 分类损失 公式3

            tmp = weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source,
                                                        torch.ones_like(domain_prob_discriminator_source))
            adv_loss = torch.mean(tmp, dim=0, keepdim=True)  # 对抗损失 公式4
            adv_loss += nn.BCELoss()(domain_prob_discriminator_target,
                                     torch.zeros_like(domain_prob_discriminator_target))

            ce_aug = nn.BCELoss(reduction='none')(predict_prob_source_aug,
                                                  one_hot(label_source, args.data.dataset.n_total))
            ce_aug = torch.sum(ce_aug) / label_source.numel()
            adv_loss_aug = nn.BCELoss()(domain_prob_source_aug, torch.ones_like(domain_prob_source_aug))  # 1
            adv_loss_aug += nn.BCELoss()(domain_prob_target_aug, torch.zeros_like(domain_prob_target_aug))  # 0

            entropy = EntropyLoss(predict_prob_target)

            # Source_Discri_weight = getattr(args.train, 'Source_Discri', 1.0)  # 如果配置文件中没有定义，则默认为1.0

            with (OptimizerManager(
                    [optimizer_cls, optimizer_discriminator, optimizer_classifier_auxiliary])):  # optimizer_finetune,
                loss = ce + args.train.adv_loss_tradeoff * adv_loss + args.train.entropy_tradeoff * entropy + \
                       args.train.adv_loss_aug_tradeoff * adv_loss_aug + args.train.ce_aug_tradeoff * ce_aug + args.train.Source_Discri_weight * Source_Discri
                # i + args.train.clip_weight * feature_alignment_loss

                # loss = ce + args.train.adv_loss_tradeoff * adv_loss + args.train.entropy_tradeoff * entropy + \
                #        args.train.adv_loss_aug_tradeoff * adv_loss_aug + args.train.ce_aug_tradeoff * ce_aug
                loss.backward()

            global_step += 1
            total_steps.update()

            if global_step % args.log.log_interval == 0:
                # print("Source_Discri_weight * Source_Discr=", args.train.Source_Discri_weight * Source_Discri)
                # print("args.train.adv_loss_aug_tradeoff * adv_loss_aug=",args.train.adv_loss_aug_tradeoff * adv_loss_aug)
                # print("args.train.ce_aug_tradeoff * ce_aug=",args.train.ce_aug_tradeoff * ce_aug)
                # print("args.train.adv_loss_tradeoff * adv_loss=",args.train.adv_loss_tradeoff * adv_loss)
                # print("args.train.entropy_tradeoff * entropy=",args.train.entropy_tradeoff * entropy)
                print("ce=", ce)
                counter = AccuracyCounter()
                counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))),
                                    variable_to_numpy(predict_prob_source))
                acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
                print('acc_train:', acc_train.item())
                logger.add_scalar('entropy', entropy, global_step)
                logger.add_scalar('adv_loss', adv_loss, global_step)
                logger.add_scalar('ce', ce, global_step)
                logger.add_scalar('adv_loss_aug', adv_loss_aug, global_step)
                logger.add_scalar('ce_aug', ce_aug, global_step)
                logger.add_scalar('acc_train', acc_train, global_step)
                logger.add_scalar('Source_Discri', Source_Discri, global_step)
                # logger.add_scalar('feature_alignment_loss', feature_alignment_loss, global_step)
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

                        counter.addOneBatch(variable_to_numpy(predict_prob),
                                            variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

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
                print("\n current_test:", acc_test)
                print("best_test:", best_acc)
                with open(join(log_dir, 'current.pkl'), 'wb') as f:
                    torch.save(data, f)

        source_class_means = source_class_mean_update.mul(1 / C_num_count)  # Class mean matrix
        source_mean = source_class_mean_update.mean(0)  # Total mean vector 域均值?
        target_Tmean = target_Tmean_update / len(target_train_ds)

    return acc_test

if __name__ == "__main__":
    import optuna

    trial_results = {}
    # Define the objective function for Optuna optimization
    def objective(trial):
        # Define the hyperparameter search space

        hyperparameters = {
            'adv_loss_tradeoff':1.0,
            'ce_aug_tradeoff': 1.0,
            'lr': 0.001,
            'entropy_tradeoff': trial.suggest_discrete_uniform('Source_Discri_weight', 0.05, 0.3, 0.1),
            'min_step': 2000,
            'adv_loss_aug_tradeoff': trial.suggest_discrete_uniform('Source_Discri_weight', 0.3, 1.0, 0.2),
            'Source_Discri_weight': trial.suggest_discrete_uniform('Source_Discri_weight', 0.2, 2.0, 0.2),
            'clip_weight': 1
        }

        print(hyperparameters)

        # Create and train the model using the hyperparameters
        totalNet = TotalNet()  # You can replace this with your model initialization
        # ... Initialize data loaders, optimizers, and other necessary components ...
        output_device = torch.device("cuda:0")
        now = datetime.datetime.now().strftime('%b%d&%H&%M&%S')

        log_dir = f'{args.log.root_dir}/{source_domain_name}&&{target_domain_name}/{now}'
        logger = SummaryWriter(log_dir)
        with open(join(log_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(save_config))

        # Train the model with the specified hyperparameters using train_model function
        acc_test = train_model(args, source_train_dl, target_train_dl, feature_extractor, classifier, discriminator,classifier_auxiliary, log_dir, output_device)

        # Store trial results
        trial_id = trial.number
        file_path = os.path.join(log_dir, f'hyperparameters_trial_{trial_id}.json')
        with open(file_path, 'w') as f:
            json.dump(hyperparameters, f)

        return acc_test  # Return the test accuracy as the optimization target

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')

    # Start the optimization process
    n_trials = 10  # You can adjust the number of trials
    study.optimize(objective, n_trials=n_trials)

    # Print the best hyperparameters and result
    best_params = study.best_params
    best_result = study.best_value
    print("Best Hyperparameters:", best_params)
    print("Best Test Accuracy:", best_result)

    trial_results['best_params'] = best_params
    trial_results['best_result'] = best_result

    # Save the trial results to a JSON file
    results_filepath = '/root/BCD_PDA/trial_results.json'  # Change this path as needed for your environment
    with open(results_filepath, 'w') as f:
        json.dump(trial_results, f)

    # H_mean_use_L1 = H_mean_update_L1.mul(1 / C_num_count)
    # H_Tmean_use_L1 = H_mean_update_L1.mean(0)
    # del source_mean