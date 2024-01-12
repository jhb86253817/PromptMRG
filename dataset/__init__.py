import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json

from .medical_dataset import generation_train, generation_eval


def create_dataset(dataset, tokenizer, args):
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(args.image_size),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    if dataset =='generation_iu_xray':
        train_dataset = generation_train(transform_train, args.image_dir, args.ann_path, tokenizer, dataset='iu_xray', args=args)
        val_dataset = generation_eval(transform_test, args.image_dir, args.ann_path, tokenizer, split='val', dataset='iu_xray', args=args)
        test_dataset = generation_eval(transform_test, args.image_dir, args.ann_path, tokenizer, split='test', dataset='iu_xray', args=args)
        return train_dataset, val_dataset, test_dataset

    elif dataset =='generation_mimic_cxr':
        train_dataset = generation_train(transform_train, args.image_dir, args.ann_path, tokenizer, dataset='mimic_cxr', args=args)
        val_dataset = generation_eval(transform_test, args.image_dir, args.ann_path, tokenizer, split='val', dataset='mimic_cxr', args=args)
        test_dataset = generation_eval(transform_test, args.image_dir, args.ann_path, tokenizer, split='test', dataset='mimic_cxr', args=args)
        return train_dataset, val_dataset, test_dataset
    
def create_dataset_test(dataset, tokenizer, args):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    if dataset =='generation_iu_xray':
        test_dataset = generation_eval(transform_test, args.image_dir, args.ann_path, tokenizer, split='test', dataset='iu_xray', args=args)
        return test_dataset
    elif dataset =='generation_mimic_cxr':
        test_dataset = generation_eval(transform_test, args.image_dir, args.ann_path, tokenizer, split='test', dataset='mimic_cxr', args=args)
        return test_dataset

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

