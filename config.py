import os
import yaml
import easydict  # easydict的作用：可以使得以属性的方式去访问字典的值！
from os.path import join
import argparse
# print("*************config开始加载**************")
# dataset
class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix             # 前缀，where are the projects
        self.domains = domains
        self.files = [(join(path, file)) for file in files]  # list of file path
        self.prefixes = [self.prefix] * len(self.domains)  # produce a list for each domains

# config in terminal

parser = argparse.ArgumentParser(description='Code for *Learning to Transfer Examples for Partial Domain Adaptation*',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()
config_file = args.config
# yaml.warnings({'YAMLLoadWarning': False})
# 加上的 因为黄色预警问题
args = yaml.load(open(config_file,'rb'),Loader=yaml.FullLoader)

save_config = yaml.load(open(config_file,'rb'),Loader=yaml.FullLoader)
# read yaml files by easydict
args = easydict.EasyDict(args)

dataset = None
# if we have default value ,above all can be ignored
if args.data.dataset.name == 'office31':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['amazon', 'dslr', 'webcam'],
    files=[
        'amazon_reorgnized.txt',
        'dslr_reorgnized.txt',
        'webcam_reorgnized.txt'
    ],
    prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'officehome':  #
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['Art', 'Clipart', 'Product', 'Real_World'],
    files=[
        'Art.txt',
        'Clipart.txt',
        'Product.txt',
        'Real_World.txt'
    ],
    prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'image_celf':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['b', 'i', 'c', 'p'],
    files=[
        'b.txt',
        'i.txt',
        'c.txt',
        'p.txt'
    ],prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'Caltech-office':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['C', 'A', 'W', 'D'],
    files=[
        'Caltech256.txt',
        'amazon_10_256.txt',
        'webcam_10_256.txt',
        'dslr_10_256.txt'
    ],prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'Caltech10-office5':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['C', 'A', 'W', 'D'],
    files=[
        'caltech10.txt',
        'amazon_10.txt',
        'webcam_10.txt',
        'dslr_10.txt'
    ],prefix=args.data.dataset.root_path)
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source_domain_name = dataset.domains[args.data.dataset.source]  # args.data.dataset.source=1
target_domain_name = dataset.domains[args.data.dataset.target]  # args.data.dataset.target=0
source_file = dataset.files[args.data.dataset.source]            #这个文件这里可能会出问题 C
target_file = dataset.files[args.data.dataset.target]#A
print('source_domain:',source_domain_name,'\ntarget_domain:',target_domain_name)
# print(source_file,target_file)
# print(dataset.prefixes[args.data.dataset.source])

# print("*************config结束加载**************")