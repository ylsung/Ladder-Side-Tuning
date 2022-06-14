import sys, os, copy
import numpy as np
import torch
import torch_pruning as tp
import transformers

import functools

def round_pruning_amount(total_parameters, n_to_prune, round_to):
    """round the parameter amount after pruning to an integer multiple of `round_to`.
    """
    n_remain = round_to*max(int(total_parameters - n_to_prune)//round_to, 1)
    return max(total_parameters - n_remain, 0)

# Load BertModel from https://github.com/huggingface/transformers

def pruning_without_residual(model, reduction_factor, importance_measure=None, version=2, num_heads=12, iterations=2):
    inputs = torch.randn(32, 3)

    # Build dependency graph
    DG = tp.DependencyGraph()
    
    # register custom pruning method for t5 layernorm
    DG.register_customized_layer(
        transformers.models.t5.modeling_t5.T5LayerNorm, 
        in_ch_pruning_fn=tp.prune_t5layernorm, # A function to prune channels/dimensions of input tensor
        out_ch_pruning_fn=tp.prune_t5layernorm, # A function to prune channels/dimensions of output tensor
        get_in_ch_fn=lambda l: None,  # estimate the n_channel of layer input. Return None if the layer does not change tensor shape.
        get_out_ch_fn=lambda l: None
    ) # estimate the n_channel of layer output. Return None if the layer does not change tensor shape.

    DG.build_dependency(model, example_inputs = inputs, pruning_dim = -1) # Note to set pruning_dim to -1 to prune BertModel on hidden_states.

    # get a pruning plan by pruning from word embedding

    if version in [2, 5]:
        strategy_for_attn = tp.strategy.L1Strategy()
        strategy_for_others = tp.strategy.L1Strategy()
    elif version == 3:
        strategy_for_attn = tp.strategy.AttnL1Strategy()
        strategy_for_others = tp.strategy.L1Strategy()
        num_heads = num_heads
        strategy_for_attn = functools.partial(strategy_for_attn, num_heads=num_heads)
    elif version == 4:
        strategy_for_attn = tp.strategy.AttnL1Strategy()
        strategy_for_others = tp.strategy.AttnL1Strategy()

        strategy_for_attn = functools.partial(strategy_for_attn, num_heads=num_heads)
        strategy_for_others = functools.partial(strategy_for_attn, num_heads=num_heads)

    if version in [2, 3, 4]:
        prune_vals = [1 - 1 / reduction_factor]
    elif version in [5]:
        prune_vals = np.linspace(0, 1 - 1 / reduction_factor, iterations)
        # print(np.rint(prune_vals * 768))
        keep_vals = 1 - prune_vals
        keep_vals_now = keep_vals[1:]
        keep_vals_pre = keep_vals[:-1] 
        prune_vals = (keep_vals_pre - keep_vals_now) / keep_vals_pre

        prune_vals = prune_vals.tolist()
        # round_pruning_amount(n, n_to_prune, round_to)

    def select_weights(weights, idxs):
        num_features = len(weights)
        keep_idxs = list(set(range(num_features)) - set(idxs))

        return weights[keep_idxs]

    state_dict = model.state_dict()
    
    if importance_measure is None:
        importance_measure = copy.deepcopy(state_dict)

    # construct ordered layers to process
    ordered_target_layers = ["layer0.weight"]

    for i in range(1, 7):
        ordered_target_layers.append([f"layer{i}.weight"])

    for prune_val in prune_vals:
        new_state_dict = {}
        for layer in ordered_target_layers:
            if isinstance(layer, list):
                # the most common case
                # will prune the weights according to previous prune idx and select next prune idx

                is_attn = any(["Attention" in sub_layer for sub_layer in layer])

                if is_attn:
                    strategy = strategy_for_attn
                else:
                    strategy = strategy_for_others
                
                weights = [state_dict[sub_layer] for sub_layer in layer]
                importances = [importance_measure[sub_layer] for sub_layer in layer]

                # prune according to previous idx
                weights = [select_weights(w.T, pruning_idxs).T for w in weights]
                importances = [select_weights(imp.T, pruning_idxs).T for imp in importances]

                prod_imp = 1
                for imp in importances:
                    prod_imp *= imp

                pruning_idxs = strategy(weights=prod_imp, amount=prune_val)

                weights = [select_weights(w, pruning_idxs) for w in weights]
                importances = [select_weights(imp, pruning_idxs) for imp in importances]

                for l, w in zip(layer, weights):
                    new_state_dict[l] = w

                # update importance measure
                for l, imp in zip(layer, importances):
                    importance_measure[l] = imp

            elif "layer0" in layer:
                # the first layer
                # will only select next prune idx
                # and direct copy the same weights
                importance = importance_measure[layer]
                weights = state_dict[layer]
                pruning_idxs = strategy_for_others(weights=importance, amount=prune_val)

                weights = select_weights(weights, pruning_idxs)
                importance = select_weights(importance, pruning_idxs)

                new_state_dict[layer] = weights
                importance_measure[layer] = importance
            
            elif "layer_norm" in layer:
                # will only prune the weights according to previous prune idx
                weights = state_dict[layer]
                weights = select_weights(weights, pruning_idxs)

                importance = importance_measure[layer]
                importance = select_weights(importance, pruning_idxs)
                
                new_state_dict[layer] = weights
                importance_measure[layer] = importance
                
            else:
                # do nothing, just copy the same weight
                new_state_dict[layer] = state_dict[layer]

        # j = 0
        # for k, v in new_state_dict.items():
        #     print(v.shape)

        #     if j > 4:
        #         break
        #     j += 1

        state_dict = new_state_dict

    return new_state_dict


def pruning_v2(model, reduction_factor, importance_measure=None):
    return pruning_without_residual(model, reduction_factor, importance_measure=importance_measure, version=2)


def pruning_v3(model, reduction_factor, importance_measure=None):
    return pruning_without_residual(model, reduction_factor, importance_measure=importance_measure, version=3, num_heads=12)


def pruning_v4(model, reduction_factor, num_heads, importance_measure=None):
    return pruning_without_residual(model, reduction_factor, importance_measure=importance_measure, version=4, num_heads=num_heads)


def pruning_v5(model, reduction_factor, importance_measure=None, iterations=11):
    return pruning_without_residual(model, reduction_factor, importance_measure=importance_measure, version=5, iterations=iterations)


if __name__ == "__main__":
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.layer0 = torch.nn.Linear(3, 64, bias=False)
            self.layer1 = torch.nn.Linear(64, 64, bias=False)
            self.layer2 = torch.nn.Linear(64, 64, bias=False)
            self.layer3 = torch.nn.Linear(64, 64, bias=False)
            self.layer4 = torch.nn.Linear(64, 64, bias=False)
            self.layer5 = torch.nn.Linear(64, 64, bias=False)
            self.layer6 = torch.nn.Linear(64, 64, bias=False)

        def forward(self, inputs):

            output = torch.nn.functional.relu(self.layer0(inputs))
            
            output = torch.nn.functional.relu(self.layer1(output))
            output = torch.nn.functional.relu(self.layer2(output))
            output = torch.nn.functional.relu(self.layer3(output))
            output = torch.nn.functional.relu(self.layer4(output))
            output = torch.nn.functional.relu(self.layer5(output))

            return self.layer6(output)

    model = Model()


    for name, _ in model.named_parameters():
        print(name)
    
    v2 = pruning_v2(model, 32)
    v5 = pruning_v5(model, 32)

    for k in v2.keys():
        print(v2[k].shape)
        print(v2[k])
        print(v5[k])

        print("==="*5)


    def get_values(d):
        v = 0

        for name, param in d.items():
            v += param.abs().sum()

        return v

    for i in [2, 3, 5, 7, 9, 11, 13, 15]:
        v = pruning_v5(model, 8, iterations=i)

        value = get_values(v)

        print(i, value)


