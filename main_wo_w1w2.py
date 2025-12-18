from data import *
from net import *
import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True

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
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)

        # self.Manifold = Manifold(256,manifold_dim[0],manifold_dim[1],classifier_output_dim)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        # w2 = self.Manifold(_)
        return y, d
totalNet = TotalNet()

# logger.add_graph(totalNet, torch.ones(2, 3, 224, 224))
feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=[0], output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=[0], output_device=output_device).train(True)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=[0], output_device=output_device).train(True)

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

# optimizer_Manifold = OptimWithSheduler(
#     optim.SGD(Manifold.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
#     scheduler)
global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0
class_weights_dict = {}
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

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_discriminator_source = discriminator.forward(feature_source)
        domain_prob_discriminator_target = discriminator.forward(feature_target)

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
            log_wei.write(f'Source_Discri is {Source_Discri.t()} \n')
        # ===================== Target discriminate Loss =========================
        if (epoch_id) <= 5:  # PLGE_step = 10  ,  PLGE_lambda_L2, PLGE_lambda_L1 = 1e1, 1e0
            Target_Discri = torch.ones_like(domain_prob_discriminator_source)
        else:  # 源域类中心H_mean_use_L2
            Target_Discri, Source_Discri_target = Target_IntraClass_sim(feature_target.detach(),feature_source.detach(),256, source_class_means,target_Tmean, label_source)
            log_wei.write(f'Source_Discri_target is {Source_Discri_target.t()} \n')
            log_wei.write(f'Target_Discri is {Target_Discri.t()} \n')

        # batch_class_weights = predict_prob_source.mean(dim=0)
        # # 每隔一定的迭代次数保存权重
        # if global_step % 500 == 0:
        #     class_weights_dict[f'Iter_{global_step}'] = batch_class_weights.cpu().detach().numpy().tolist()
        #     predict_prob_source(class_weights_dict, log_dir + 'class_weights.json')

        # ============================== cross entropy loss, it receives logits as its inputs
        ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source).view(-1, 1)
        ce = torch.mean(ce, dim=0, keepdim=True)

        tmp = nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss = torch.mean(tmp, dim=0, keepdim=True)
        adv_loss += nn.BCELoss()(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))


        entropy = EntropyLoss(predict_prob_target)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_discriminator]):
            loss = ce + args.train.adv_loss_tradeoff * adv_loss + args.train.entropy_tradeoff * entropy


            loss.backward()

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
            print('acc_train:',acc_train.item())
            logger.add_scalar('entropy', entropy, global_step)
            logger.add_scalar('adv_loss', adv_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)
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
