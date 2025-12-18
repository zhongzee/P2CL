"""
P2CL: Prototype-Constrained Consistent Learning
Toward Controllable and Consistent Transfer for Partial Domain Adaptation

This implementation includes:
- CPT (Controllable Prototype-guided Transfer): Eq. 4-8
- DCC (Discrepancy-gated Consistency Calibration): Eq. 9-10
- Overall objective: Eq. 11
"""

from data import *
from net import *
import datetime
from tqdm import tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True
from easydl import inverseDecaySheduler, OptimWithSheduler, OptimizerManager, one_hot, TorchLeakySoftmax
import torch.nn as nn
import torch.nn.functional as F
import json

output_device = torch.device("cuda:0")

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc,
    'alexnet': AlexNetFc
}


class TotalNet(nn.Module):
    def __init__(self, n_classes):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = n_classes
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        # Auxiliary classifier C1 for dual classifier (Eq. 8)
        self.classifier_auxiliary = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, classifier_output_dim),
        )

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        y_aug = self.classifier_auxiliary(_)
        return y, d, y_aug


class PrototypeMemoryBank:
    """
    Momentum-updated Prototype Memory Bank (Eq. 5)
    """
    def __init__(self, n_classes, feature_dim, momentum=0.9, device='cuda'):
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.device = device
        # Initialize prototypes with zeros (will be set on first update)
        self.prototypes = torch.zeros(n_classes, feature_dim).to(device)
        self.initialized = torch.zeros(n_classes, dtype=torch.bool).to(device)
    
    def update(self, features, labels):
        """
        Update prototypes with EMA (Eq. 5)
        t_c^(k+1) <- norm(mu * t_c^(k) + (1-mu) * f_bar_c^s)
        """
        features = features.detach()
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        for c in range(self.n_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_features = features_norm[mask].mean(dim=0)
                if self.initialized[c]:
                    # EMA update
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * class_features
                else:
                    # First time initialization
                    self.prototypes[c] = class_features
                    self.initialized[c] = True
                # Re-normalize prototype
                self.prototypes[c] = F.normalize(self.prototypes[c], p=2, dim=0)
    
    def get_prototypes(self):
        """Return normalized prototypes for shared classes"""
        return F.normalize(self.prototypes, p=2, dim=1)


def compute_cpt_weight(domain_prob, features, prototype_bank, n_share):
    """
    Controllable Prototype-guided Transfer (CPT) - Section 3.1
    
    Args:
        domain_prob: D(x) from discriminator, shape [batch_size, 1]
        features: normalized features, shape [batch_size, feature_dim]
        prototype_bank: PrototypeMemoryBank instance
        n_share: number of shared classes |S|
    
    Returns:
        w(x): controllable weight for each sample
    """
    # Sample-wise transferability gate (Eq. 4)
    # g(x) = 1 - |2D(x) - 1|
    g = 1.0 - torch.abs(2.0 * domain_prob - 1.0)  # [batch_size, 1]
    
    # Get shared class prototypes [n_share, feature_dim]
    prototypes = prototype_bank.get_prototypes()[:n_share]
    
    # Normalize features
    features_norm = F.normalize(features, p=2, dim=1)  # [batch_size, feature_dim]
    
    # Compute cosine similarity to shared prototypes (Eq. 6)
    # r_S(x) = [t_c^T * f_hat(x)]_{c in S}
    cos_sim = torch.mm(features_norm, prototypes.t())  # [batch_size, n_share]
    
    # Prototype distribution (Eq. 6)
    s = F.softmax(cos_sim, dim=1)  # [batch_size, n_share]
    
    # Prototype focus (Eq. 7)
    # kappa(x) = max_{c in S} s_c(x)
    kappa = s.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
    
    # Controllable weight (Eq. 8) with batch normalization
    # w(x) = BN(g * kappa)
    raw_weight = g * kappa  # [batch_size, 1]
    
    # Batch-wise normalization: w(x) = (g*kappa) / mean(g*kappa)
    w = raw_weight / (raw_weight.mean() + 1e-8)
    
    return w, s, kappa, g


def compute_dcc_loss(classifier_logits, prototype_dist, cpt_weight, n_share, n_total):
    """
    Discrepancy-gated Consistency Calibration (DCC) - Section 3.2
    
    Args:
        classifier_logits: raw logits from classifier, shape [batch_size, n_total]
        prototype_dist: s(x) from CPT, shape [batch_size, n_share]
        cpt_weight: w(x) from CPT, shape [batch_size, 1]
        n_share: number of shared classes |S|
        n_total: total number of classes
    
    Returns:
        L_DCC: Discrepancy-gated consistency loss (Eq. 10)
    """
    # Get classifier distribution restricted to shared classes and renormalized
    # p_{|S}(x)
    p_full = F.softmax(classifier_logits, dim=1)  # [batch_size, n_total]
    p_shared = p_full[:, :n_share]  # [batch_size, n_share]
    p_shared_norm = p_shared / (p_shared.sum(dim=1, keepdim=True) + 1e-8)  # renormalize
    
    # Discrepancy measurement using JS divergence (Eq. 9)
    # d(x) = JS(p_{|S}(x) || s(x))
    m = 0.5 * (p_shared_norm + prototype_dist)  # mixture distribution
    
    # JS divergence = 0.5 * KL(p||m) + 0.5 * KL(s||m)
    kl_p_m = F.kl_div(torch.log(m + 1e-8), p_shared_norm, reduction='none').sum(dim=1)
    kl_s_m = F.kl_div(torch.log(m + 1e-8), prototype_dist, reduction='none').sum(dim=1)
    d = 0.5 * (kl_p_m + kl_s_m)  # [batch_size]
    
    # Entropy of p_{|S}(x)
    H_p = -torch.sum(p_shared_norm * torch.log(p_shared_norm + 1e-8), dim=1)  # [batch_size]
    max_entropy = np.log(n_share)
    
    # Confidence suppression factor (Eq. 10)
    # alpha = 1 - H(p_{|S}) / log|S|
    alpha = 1.0 - H_p / max_entropy  # [batch_size]
    
    # DCC loss (Eq. 10)
    # L_DCC = (1/n_t) * sum_j w(x_j^t) * d(x_j^t) * alpha(x_j^t)
    L_dcc = (cpt_weight.squeeze() * d * alpha).mean()
    
    return L_dcc, d.mean()


def compute_weighted_cls_loss(logits_main, logits_aux, labels, weights):
    """
    Weighted classification loss with dual classifiers (Eq. 8)
    L_cls^w = (1/n_s) * sum_i w(x_i^s) * [CE(C(F(x_i^s)), y_i^s) + CE(C1(F(x_i^s)), y_i^s)]
    """
    ce_main = F.cross_entropy(logits_main, labels, reduction='none')
    ce_aux = F.cross_entropy(logits_aux, labels, reduction='none')
    
    weighted_loss = (weights.squeeze() * (ce_main + ce_aux)).mean()
    return weighted_loss


def compute_weighted_adv_loss(domain_prob_source, domain_prob_target, weights_source, weights_target):
    """
    Weighted adversarial loss (Eq. 9 in paper, implemented as standard adversarial loss with CPT weights)
    """
    # Source domain: label = 1
    adv_source = -weights_source.squeeze() * torch.log(domain_prob_source.squeeze() + 1e-8)
    # Target domain: label = 0
    adv_target = -weights_target.squeeze() * torch.log(1 - domain_prob_target.squeeze() + 1e-8)
    
    return adv_source.mean() + adv_target.mean()


def train_model(args, source_train_dl, target_train_dl, log_dir, output_device):
    """
    Main training loop for P2CL
    """
    logger = SummaryWriter(log_dir)
    log_text = open(join(log_dir, 'log.txt'), 'w')
    log_train = open(join(log_dir, 'log_train.txt'), 'w')
    
    seed_everything()
    
    n_share = args.data.dataset.n_share
    n_total = args.data.dataset.n_total
    feature_dim = 256  # bottleneck dimension
    
    # Initialize network
    totalNet = TotalNet(n_total)
    feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=[0], output_device=output_device).train(True)
    classifier = nn.DataParallel(totalNet.classifier, device_ids=[0], output_device=output_device).train(True)
    discriminator = nn.DataParallel(totalNet.discriminator, device_ids=[0], output_device=output_device).train(True)
    classifier_auxiliary = nn.DataParallel(totalNet.classifier_auxiliary, device_ids=[0], output_device=output_device).train(True)
    
    # Initialize Prototype Memory Bank
    prototype_bank = PrototypeMemoryBank(
        n_classes=n_total,
        feature_dim=feature_dim,
        momentum=0.9,  # mu in paper
        device=output_device
    )
    
    # Optimizers
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
    optimizer_finetune = OptimWithSheduler(
        optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, 
                  weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_cls = OptimWithSheduler(
        optim.SGD(classifier.parameters(), lr=args.train.lr, 
                  weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_discriminator = OptimWithSheduler(
        optim.SGD(discriminator.parameters(), lr=args.train.lr, 
                  weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    optimizer_classifier_auxiliary = OptimWithSheduler(
        optim.SGD(classifier_auxiliary.parameters(), lr=args.train.lr / 10.0, 
                  weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
        scheduler)
    
    # Hyperparameters (Eq. 11): L = alpha*L_cls^w + beta*L_adv^w + lambda*L_DCC
    alpha = getattr(args.train, 'alpha', 2.0)
    beta = getattr(args.train, 'beta', 1.5)
    lambda_dcc = getattr(args.train, 'lambda_dcc', 1.2)
    
    total_steps = tqdm(range(args.train.min_step), desc='global step')
    global_step = 0
    best_acc = 0
    epoch_id = 0
    
    while global_step < args.train.min_step:
        iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id}',
                     total=min(len(source_train_dl), len(target_train_dl)))
        epoch_id += 1
        
        for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):
            label_source = label_source.to(output_device)
            im_source = im_source.to(output_device)
            im_target = im_target.to(output_device)
            
            # Forward pass - Source
            fc1_s = feature_extractor.forward(im_source)
            fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
            domain_prob_source = discriminator.forward(feature_source)
            logits_aux_source = classifier_auxiliary.forward(feature_source)
            
            # Forward pass - Target
            fc1_t = feature_extractor.forward(im_target)
            fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)
            domain_prob_target = discriminator.forward(feature_target)
            
            # Update Prototype Memory Bank with source samples (Eq. 5)
            prototype_bank.update(feature_source, label_source)
            
            # Compute CPT weights for source and target (Section 3.1)
            w_source, s_source, kappa_source, g_source = compute_cpt_weight(
                domain_prob_source, feature_source, prototype_bank, n_share)
            w_target, s_target, kappa_target, g_target = compute_cpt_weight(
                domain_prob_target, feature_target, prototype_bank, n_share)
            
            # Weighted Classification Loss (Eq. 8)
            L_cls_w = compute_weighted_cls_loss(fc2_s, logits_aux_source, label_source, w_source)
            
            # Weighted Adversarial Loss (Eq. 9)
            L_adv_w = compute_weighted_adv_loss(domain_prob_source, domain_prob_target, 
                                                 w_source.detach(), w_target.detach())
            
            # DCC Loss on target samples (Eq. 10)
            L_dcc, mean_discrepancy = compute_dcc_loss(fc2_t, s_target, w_target, n_share, n_total)
            
            # Total loss (Eq. 11)
            loss = alpha * L_cls_w + beta * L_adv_w + lambda_dcc * L_dcc
            
            # Optimization
            with OptimizerManager([optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_classifier_auxiliary]):
                loss.backward()
            
            global_step += 1
            total_steps.update()
            
            # Logging
            if global_step % args.log.log_interval == 0:
                counter = AccuracyCounter()
                counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))),
                                    variable_to_numpy(predict_prob_source))
                acc_train = counter.reportAccuracy()
                
                logger.add_scalar('L_cls_w', L_cls_w.item(), global_step)
                logger.add_scalar('L_adv_w', L_adv_w.item(), global_step)
                logger.add_scalar('L_dcc', L_dcc.item(), global_step)
                logger.add_scalar('total_loss', loss.item(), global_step)
                logger.add_scalar('acc_train', acc_train, global_step)
                logger.add_scalar('mean_gate_g', g_target.mean().item(), global_step)
                logger.add_scalar('mean_kappa', kappa_target.mean().item(), global_step)
                logger.add_scalar('mean_discrepancy', mean_discrepancy.item(), global_step)
                
                log_train.write(f'epoch:{epoch_id} step:{global_step} | '
                               f'L_cls={L_cls_w.item():.4f} L_adv={L_adv_w.item():.4f} '
                               f'L_dcc={L_dcc.item():.4f} acc={acc_train:.4f}\n')
                log_train.flush()
                print(f"Step {global_step}: L_cls={L_cls_w.item():.4f}, L_adv={L_adv_w.item():.4f}, "
                      f"L_dcc={L_dcc.item():.4f}, acc_train={acc_train:.4f}")
            
            # Testing
            if global_step % args.test.test_interval == 0:
                counter = AccuracyCounter()
                with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
                    for im, label in target_train_dl:
                        im = im.to(output_device)
                        label = label.to(output_device)
                        feature = feature_extractor.forward(im)
                        ___, __, before_softmax, predict_prob = classifier.forward(feature)
                        counter.addOneBatch(variable_to_numpy(predict_prob),
                                            variable_to_numpy(one_hot(label, n_total)))
                
                acc_test = counter.reportAccuracy()
                logger.add_scalar('acc_test', acc_test, global_step)
                log_text.write(f'step:{global_step} acc_test={acc_test:.4f}\n')
                log_text.flush()
                
                # Save model
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
                
                print(f"\nTest accuracy: {acc_test:.4f}, Best: {best_acc:.4f}")
                
                with open(join(log_dir, 'current.pkl'), 'wb') as f:
                    torch.save(data, f)
    
    log_text.close()
    log_train.close()
    return best_acc


if __name__ == "__main__":
    now = datetime.datetime.now().strftime('%b%d_%H%M%S')
    log_dir = f'{args.log.root_dir}/{source_domain_name}_{target_domain_name}/{now}'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = SummaryWriter(log_dir)
    with open(join(log_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(save_config))
    
    print("=" * 60)
    print("P2CL: Prototype-Constrained Consistent Learning")
    print("=" * 60)
    print(f"Source: {source_domain_name} -> Target: {target_domain_name}")
    print(f"Shared classes: {args.data.dataset.n_share}, Total classes: {args.data.dataset.n_total}")
    print(f"Log directory: {log_dir}")
    print("=" * 60)
    
    best_acc = train_model(args, source_train_dl, target_train_dl, log_dir, output_device)
    
    print(f"\nTraining completed. Best accuracy: {best_acc:.4f}")
    
    # Save final results
    results = {
        'source': source_domain_name,
        'target': target_domain_name,
        'best_accuracy': best_acc,
        'n_share': args.data.dataset.n_share,
        'n_total': args.data.dataset.n_total,
    }
    with open(join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
