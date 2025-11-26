import math

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler


class DistBalancedBatchSampler(BatchSampler):
    """
    dataset: dataset to be sampled
    num_classes : number of classes in the dataset
    n_sample_classes : the number of classes to be sampled in one batch
    n_samples: the number of samples to be sampled for each class in *n_sample_classes*
    seed: use the same seed for each replica
    num_replicas:
    rank:
    """

    def __init__(self, dataset, num_classes, n_sample_classes, n_samples, seed=666, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.rank = rank
        self.labels = torch.LongTensor([ann['label'] for ann in dataset.annotations])

        self.labels_set = list(np.arange(num_classes))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            # use the same seed for each replica
            np.random.seed(self.seed)
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_sample_classes = n_sample_classes
        self.n_samples = n_samples

        # batch_size refers to bs per replica
        self.batch_size = self.n_samples * self.n_sample_classes

        # for the whole data set, each replica should sample `total_samples_per_replica` samples.
        self.total_samples_per_replica = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

    def __iter__(self):
        # self.count for each replica
        self.count = 0
        while self.count + self.batch_size < self.total_samples_per_replica:
            classes = np.random.choice(self.labels_set, self.n_sample_classes, replace=False)
            indices = []
            for class_ in classes:
                start = self.used_label_indices_count[class_] + (self.rank % self.num_replicas)
                end = self.used_label_indices_count[class_] + self.n_samples * self.num_replicas
                step = self.num_replicas
                indices.extend(self.label_to_indices[class_][start:end:step])
                self.used_label_indices_count[class_] += self.n_samples * self.num_replicas
                if self.used_label_indices_count[class_] + self.n_samples * self.num_replicas > len(self.label_to_indices[class_]):
                    np.random.seed(self.seed)
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            #  print(f'{self.rank} indices:{indices}.')
            yield from indices
            self.count += self.n_sample_classes * self.n_samples

    def __len__(self):
        return self.total_samples_per_replica
