import torch
from abc import abstractclassmethod, ABC
from typing import Sequence
import random
import warnings
import math

# https://github.com/VainF/Torch-Pruning/issues/49 by @Serjio42
def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """
    n_remain = round_to*max(int(total_parameters - n_to_prune)//round_to, 1)
    return max(total_parameters - n_remain, 0)

class BaseStrategy(ABC):
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abstractclassmethod
    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        """ Apply the strategy on weights with user specified pruning percentage.

        Parameters:
            weights (torch.Parameter): weights to be pruned.
            amount (Callable): the percentage of weights to be pruned (amount<1.0) or the amount of weights to be pruned (amount>=1.0) 
            round_to (int): the number to which the number of pruned channels is rounded.
        """
        raise NotImplementedError

class RandomStrategy(BaseStrategy):

    def apply(self, weights, amount=0.0, round_to=1)->  Sequence[int]:  # return index
        if amount<=0: return []
        n = len(weights)
        n_to_prune = int(amount*n) if amount<1.0 else amount
        n_to_prune = round_pruning_amount(n, n_to_prune, round_to)
        if n_to_prune == 0: return []
        indices = random.sample( list( range(n) ), k=n_to_prune )
        return indices

class LNStrategy(BaseStrategy):
    def __init__(self, p):
        self.p = p

    def apply(self, weights, amount=0.0, round_to=1, n_to_prune=-1)->  Sequence[int]:  # return index
        if n_to_prune == -1:
            if amount<=0: return []
            n = len(weights)
            l1_norm = torch.norm( weights.view(n, -1), p=self.p, dim=1 )
            n_to_prune = round(amount*n) if amount<1.0 else amount 
            n_to_prune = round_pruning_amount(n, n_to_prune, round_to)

        if n_to_prune == 0: return []
        # threshold = torch.kthvalue(l1_norm, k=n_to_prune).values 
        # indices = torch.nonzero(l1_norm <= threshold).view(-1).tolist()
        indices = torch.sort(torch.argsort(l1_norm)[:n_to_prune])[0].tolist()

        # print(len(new_indices), len(indices))

        # print(new_indices, indices)
        # 2903, 2902
        # for i in range(min(len(new_indices), len(indices))):
        #     # print(new_indices[i], indices[i])

        #     try:
        #         assert new_indices[i] == indices[i]
        #     except:

        #         print(len(torch.nonzero(l1_norm <= l1_norm[3071]).view(-1).tolist()))
        #         print(l1_norm[3071], l1_norm[2902], threshold, n_to_prune)
        #         print(new_indices[i], indices[i])

        return indices


class L1Strategy(LNStrategy):
    def __init__(self):
        super(L1Strategy, self).__init__(p=1)

class L2Strategy(LNStrategy):
    def __init__(self):
        super(L2Strategy, self).__init__(p=2)


class AttnLNStrategy(BaseStrategy):
    def __init__(self, p):
        self.p = p

    def apply(self, weights, num_heads, amount=0.0, round_to=1, n_to_prune=-1)->  Sequence[int]:  # return index
        if n_to_prune == -1:
            if amount<=0: return []
            n = len(weights)
            dim_per_head = n // num_heads
            l1_norm = torch.norm( weights.reshape(num_heads, -1), p=self.p, dim=1 )
            
            n_to_prune = round(amount*n) if amount<1.0 else amount 
            n_to_prune = round_pruning_amount(n, n_to_prune, round_to)

        if n_to_prune == 0: return []

        n_to_keep = n - n_to_prune
        n_head_to_keep = math.ceil(n_to_keep / dim_per_head)
        indices = torch.argsort(l1_norm, descending=True)[:n_head_to_keep].tolist()

        keep_indices = []
        n_to_keep_left = n_to_keep
        for idx in indices:
            n_this_batch = min(n_to_keep_left, dim_per_head)

            sub_weights = weights[idx * dim_per_head: (idx + 1) * dim_per_head]

            l1_norm = torch.norm( sub_weights, p=self.p, dim=1 )

            sub_indices = torch.argsort(l1_norm, descending=True)[:n_this_batch].tolist()

            sub_indices = [s + idx * dim_per_head for s in sub_indices]

            keep_indices += sub_indices

            n_to_keep_left -= n_this_batch

        indices = list(set(list(range(n))) - set(keep_indices))

        return indices


class AttnL1Strategy(AttnLNStrategy):
    def __init__(self):
        super(AttnL1Strategy, self).__init__(p=1)


class AttnL2Strategy(AttnLNStrategy):
    def __init__(self):
        super(AttnL2Strategy, self).__init__(p=2)
