import copy
import os

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.dataloader import default_collate

from utils.sampler import DistBalancedBatchSampler
from .dataset import GEBDDataset


def build_dataloader(cfg, args, dataset_splits, is_train):
    assert len(dataset_splits) >= 1

    def build_dataset(dataset):
        name, split = dataset.split('_')

        ROOT = {
            'GEBD': os.getenv('GEBD_ROOT', 'data/Kinetics-GEBD/images'),
            'TAPOS': os.getenv('TAPOS_ROOT', 'data/TAPOS/images'),
        }

        datasets = {
            'GEBD': {
                'train': 'train',
                'val': 'val',
                'minval': 'val',
                'val_minus_minval': 'GEBD_val_frames',
                'test': 'GEBD_test_frames',
            },
            'TAPOS': {
                'train': 'train',
                'val': 'val'
            }
        }

        assert name in datasets, f'Dataset {name} not exists!'
        root = os.path.join(ROOT[name], datasets[name][split])
        template = 'frame{:d}.jpg'

        dataset = GEBDDataset(cfg, root=root,
                              name=name,
                              split=split,
                              template=template,
                              train=is_train)
        # if is_train==True:
        #     print(len(dataset))
        #     assert 1==2                     
        return dataset
        
    datasets = []
    for split in dataset_splits:
        datasets.append(build_dataset(split))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        annotations = []
        for dataset in datasets:
            annotations.extend(copy.deepcopy(dataset.annotations))
        dataset = ConcatDataset(datasets)
        dataset.annotations = annotations

    if args.distributed:
        if is_train and not cfg.INPUT.END_TO_END:
            sampler = DistBalancedBatchSampler(dataset, num_classes=2, n_sample_classes=2, n_samples=cfg.SOLVER.BATCH_SIZE // 2)
        else:
            sampler = DistributedSampler(dataset, shuffle=is_train)
    else:
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)

    # collate_fn = (lambda x: x) if cfg.INPUT.END_TO_END else default_collate
    collate_fn = default_collate
    loader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                        sampler=sampler,
                        drop_last=False,
                        collate_fn=collate_fn,
                        num_workers=cfg.SOLVER.NUM_WORKERS,
                        pin_memory=True)
    return loader
