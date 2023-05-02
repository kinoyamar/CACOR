import torch
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.nn.utils.rnn import pack_sequence
import random

from clutils.datasets.utils import unpack_sequences, ListDataset, collate_sequences_pack

class Rehearsal():
    def __init__(self, patterns_per_class, patterns_per_class_per_batch=0):
        """
        :param patterns_per_class_per_batch:
            if <0 -> concatenate patterns to the entire dataloader
            if 0 -> concatenate to the current batch another batch size
                    split among existing classes
            if >0 -> concatenate to the current batch `patterns_per_class_per_batch`
                    patterns for each existing class
        """

        self.patterns_per_class = patterns_per_class
        self.patterns_per_class_per_batch = patterns_per_class_per_batch
        self.add_every_batch = patterns_per_class_per_batch >= 0

        self.patterns = {}

    def record_patterns(self, dataloader):
        """
        Update rehearsed patterns with the current data
        """

        counter = defaultdict(int)
        for x,y in dataloader:
            # loop over each minibatch
            for el, _t in zip(x,y):
                t = _t.item()
                if t not in self.patterns:
                    self.patterns[t] = el.unsqueeze(0).clone()
                    counter[t] += 1
                elif counter[t] < self.patterns_per_class:
                    self.patterns[t] = torch.cat( (self.patterns[t], el.unsqueeze(0).clone()) )
                    counter[t] += 1
    

    def concat_to_batch(self, x,y):
        """
        Concatenate subset of memory to the current batch.
        """
        if not self.add_every_batch or self.patterns == {}:
            return x, y

        # how many replay patterns per class per batch?
        # either patterns_per_class_per_batch
        # or batch_size split equally among existing classes
        to_add = int(y.size(0) / len(self.patterns.keys())) \
                if self.patterns_per_class_per_batch == 0 \
                else self.patterns_per_class_per_batch

        rehe_x, rehe_y = [x], [y]
        for k,v in self.patterns.items():
            if to_add >= v.size(0):
                # take directly the memory
                rehe_x.append(v)
            else:
                # select at random from memory
                subset = v[torch.randperm(v.size(0))][:to_add]
                rehe_x.append(subset)
            rehe_y.append(torch.ones(rehe_x[-1].size(0)).long() * k)

        return torch.cat(rehe_x, dim=0), torch.cat(rehe_y, dim=0)


    def _tensorize(self):
        """
        Put the rehearsed pattern into a TensorDataset
        """

        x = []
        y = []
        for k, v in self.patterns.items():
            x.append(v)
            y.append(torch.ones(v.size(0)).long() * k)

        x, y = torch.cat(x), torch.cat(y)

        return TensorDataset(x, y)

    def augment_dataset(self, dataloader, shuffle=True, drop_last=True):
        """
        Add rehearsed pattern to current dataloader
        """
        if self.add_every_batch or self.patterns == {}:
            return dataloader
        else:
            return DataLoader( ConcatDataset((
                    dataloader.dataset, 
                    self._tensorize()
                )), shuffle=shuffle, drop_last=drop_last, batch_size=dataloader.batch_size)

    def create_memory_generator(self, batch_size=128):
        def memory_generator():
            memory_dataloader = DataLoader(self._tensorize(), shuffle=True, drop_last=True, batch_size=batch_size)
            yield
            while True:
                for x, y in memory_dataloader:
                    yield x, y
        gen = memory_generator()
        gen.__next__()
        return gen.__next__

    def memory_loader(self, batch_size=128, shuffle=True, drop_last=True):
        return DataLoader(self._tensorize(), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class PackedRehearsal(Rehearsal):
    """
    Rehearsal that deals with PackedSequences
    """

    def record_patterns(self, dataloader):
        """
        Update rehearsed patterns with the current data
        """

        counter = defaultdict(int)
        for x,y in dataloader:
            x = unpack_sequences(x)
            # loop over each minibatch
            for el, _t in zip(x,y):
                t = _t.item()
                if t not in self.patterns:
                    self.patterns[t] = [el.clone()]
                    counter[t] += 1
                elif counter[t] < self.patterns_per_class:
                    self.patterns[t].append(el.clone())
                    counter[t] += 1

    def concat_to_batch(self, x,y):
        """
        Concatenate subset of memory to the current batch.
        """
        if not self.add_every_batch or self.patterns == {}:
            return x, y

        # how many replay patterns per class per batch?
        # either patterns_per_class_per_batch
        # or batch_size split equally among existing classes
        to_add = int(y.size(0) / len(self.patterns.keys())) \
                if self.patterns_per_class_per_batch == 0 \
                else self.patterns_per_class_per_batch

        x = unpack_sequences(x)
        rehe_x, rehe_y = [*x], [y]
        for k,v in self.patterns.items():
            if to_add >= len(v):
                # take directly the memory
                rehe_x.extend(v)
                rehe_y.append(torch.ones(len(v)).long() * k)
            else:
                # select at random from memory
                subset = random.sample(v, to_add)
                rehe_x.extend(subset)
                rehe_y.append(torch.ones(to_add).long() * k)

        return pack_sequence(rehe_x, enforce_sorted=False), torch.cat(rehe_y, dim=0)

    def _to_dataset(self):
        """
        Put the rehearsed pattern into a Dataset
        """
        x = []
        y = []
        for k, v in self.patterns.items():
            x.extend(v)
            y.append(torch.ones(len(v)).long() * k)

        y = torch.cat(y)

        return ListDataset(x, y)

    def augment_dataset(self, dataloader, shuffle=True, drop_last=True):
        """
        Add rehearsed pattern to current dataloader
        """
        if self.add_every_batch or self.patterns == {}:
            return dataloader
        else:
            return DataLoader( ConcatDataset((
                    dataloader.dataset, 
                    self._to_dataset()
                )), shuffle=shuffle, drop_last=drop_last, batch_size=dataloader.batch_size, collate_fn=collate_sequences_pack)

    def memory_loader(self, batch_size=128, shuffle=True, drop_last=True):
        return DataLoader(self._to_dataset(), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate_sequences_pack)
