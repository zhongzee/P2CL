# P2CL: Prototype-Constrained Consistent Learning

**Toward Controllable and Consistent Transfer for Partial Domain Adaptation**

*ICASSP 2026 Submission*

## Overview

P2CL addresses two critical challenges in Partial Domain Adaptation (PDA):
1. **Insufficient controllability** over transfer selection under label asymmetry
2. **Miscalibration** between geometric similarity and classifier logits

Our framework establishes bilateral control over sample-level transferability and view-level consistency through two core modules:
- **CPT (Controllable Prototype-guided Transfer)**: Controls what and how to align
- **DCC (Discrepancy-gated Consistency Calibration)**: Aligns multi-view signals

## Method

### CPT (Controllable Prototype-guided Transfer)

1. **Sample-wise Transferability Gate** (Eq. 4):
   ```
   g(x) = 1 - |2D(x) - 1|
   ```
   where D(x) is the domain discriminator output.

2. **Prototype Memory Bank** with EMA update (Eq. 5):
   ```
   t_c^(k+1) = norm(μ·t_c^(k) + (1-μ)·f̄_c^s)
   ```

3. **Prototype Focus** (Eq. 6-7):
   ```
   s(x) = softmax(cos(f(x), T))
   κ(x) = max_{c∈S} s_c(x)
   ```

4. **Controllable Weight** (Eq. 8):
   ```
   w(x) = BN(g(x) · κ(x))
   ```

### DCC (Discrepancy-gated Consistency Calibration)

1. **Discrepancy Measurement** using JS divergence (Eq. 9):
   ```
   d(x) = JS(p_{|S}(x) || s(x))
   ```

2. **DCC Loss** with confidence suppression (Eq. 10):
   ```
   L_DCC = w(x) · d(x) · (1 - H(p_{|S})/log|S|)
   ```

### Overall Objective (Eq. 11)

```
L = α·L_cls^w + β·L_adv^w + λ·L_DCC
```

## Requirements

- Python 3.7+
- PyTorch >= 1.7.0
- torchvision
- tensorboardX
- tqdm
- numpy
- easydl

## Installation

```bash
pip install torch torchvision tensorboardX tqdm numpy
pip install easydl
```

## Datasets

Download and organize datasets:

- **Office-31**: [Link](https://faculty.cc.gatech.edu/~judy/domainadapt/)
- **ImageNet-Caltech**: Subset of ImageNet and Caltech-256
- **VisDA-2017**: [Link](http://ai.bu.edu/visda-2017/)

## Usage

### Training

```bash
# Office-31 dataset
python main_opti_CLIP_P2CL.py --config config_office31.yaml

# ImageNet-Caltech dataset
python main_opti_CLIP_P2CL.py --config config_caltech_office31_clip.yaml
```

### Configuration

Key hyperparameters in config files:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha` | Classification loss weight | 2.0 |
| `beta` | Adversarial loss weight | 1.5 |
| `lambda_dcc` | DCC loss weight | 1.2 |
| `momentum` (μ) | Prototype EMA momentum | 0.9 |
| `lr` | Learning rate | 0.001 |
| `min_step` | Total training iterations | 1000 |

## Project Structure

```
P2CL/
├── main_opti_CLIP_P2CL.py    # Main training script (P2CL implementation)
├── net.py                     # Network architectures
├── data.py                    # Data loading utilities
├── config_office31.yaml       # Office-31 config
├── config_caltech_office31_clip.yaml  # ImageNet-Caltech config
├── datasets/                  # Dataset directory
├── log/                       # Training logs
└── README.md
```

## Results

### Office-31 (ResNet-50)

| Method | A→W | D→W | W→D | A→D | D→A | W→A | Avg |
|--------|-----|-----|-----|-----|-----|-----|-----|
| P2CL | 97.25 | 100.0 | 100.0 | 98.73 | 96.36 | 96.92 | **98.21** |

### ImageNet-Caltech

| Method | I→C | C→I | Avg |
|--------|-----|-----|-----|
| P2CL | 82.6 | 80.8 | **81.7** |

### VisDA-2017

| Method | R→S | S→R | Avg |
|--------|-----|-----|-----|
| P2CL | 72.3 | 75.9 | **74.1** |

## Ablation Study

| +CPT | +DCC | ImageNet-Caltech | Office-31 | VisDA2017 | Avg |
|------|------|------------------|-----------|-----------|-----|
| ✗ | ✗ | 79.12 | 96.73 | 74.53 | 83.46 |
| ✓ | ✗ | 81.75 | 97.95 | 78.34 | 86.01 |
| ✗ | ✓ | 81.12 | 97.58 | 77.46 | 85.39 |
| ✓ | ✓ | **82.06** | **98.21** | **78.92** | **86.39** |

## Citation

```bibtex
@misc{p2cl2026,
  title={P2CL: Prototype-Constrained Consistent Learning --- Toward Controllable and Consistent Transfer},
  author={Long, Yitian and Wu, Zhongze and Yang, Feng and You, Shan and Xu, Hongyan and Su, Xiu},
  year={2025},
  note={Under review}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

This work builds upon the ETN framework and incorporates ideas from prototype-based domain adaptation methods.
