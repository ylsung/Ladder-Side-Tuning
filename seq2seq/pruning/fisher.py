import torch
import datasets
import inspect
from packaging import version
from torch.utils.data import DataLoader


def remove_unused_columns(model, dataset, description=None):
    _signature_columns = None
    if _signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(model.forward)
        _signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        _signature_columns += ["label", "label_ids"]

    ignored_columns = list(set(dataset.column_names) - set(_signature_columns))

    columns = [k for k in _signature_columns if k in dataset.column_names]

    if version.parse(datasets.__version__) < version.parse("1.4.0"):
        dataset.set_format(
            type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
        )
        return dataset
    else:
        return dataset.remove_columns(ignored_columns)


def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        loss = return_dicts["loss"]

        loss.backward()

        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data
        
        model.zero_grad()

    return gradients_dict


def calculate_the_importance_expect(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        logits = return_dicts["logits"]

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data

                model.zero_grad()

    return gradients_dict


def compute_fisher(model, train_dataset, data_collator, num_samples):
    importance_method = calculate_the_importance_label

    # import pdb
    # pdb.set_trace()

    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    train_dataset = remove_unused_columns(model, train_dataset, description="training")

    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True
    )
    
    grad_type = "square"

    return importance_method(model, data_loader, num_samples, cuda_device, grad_type)


def create_mask_gradient(model, train_dataset, data_collator, num_samples, keep_ratio, sample_type, grad_type):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True
    )

    if sample_type == "label":
        importance_method = calculate_the_importance_label
    elif sample_type == "expect":
        importance_method = calculate_the_importance_expect
    else:
        raise NotImplementedError

    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=cuda_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict


def create_mask_random(model, train_dataset, data_collator, num_samples, keep_ratio):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = torch.rand(param.shape).to(original_device)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=original_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(pretrain_weight_size, classifier_size, all_params_size)
    print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

    return mask_dict


def create_mask_bias(model, train_dataset, data_collator, num_samples, keep_ratio):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    mask_dict = {}

    for name, param in model.named_parameters():
        if "classifier" in name:
            mask_dict[name] = torch.ones_like(param, device=original_device)
        elif "bias" in name:
            mask_dict[name] = torch.ones_like(param, device=original_device)
        else:
            mask_dict[name] = torch.zeros_like(param, device=original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    bias_params_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            bias_params_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    print(bias_params_size, classifier_size, all_params_size)

    print(f"trainable parameters: {(bias_params_size + classifier_size) / all_params_size * 100} %")

    model.to(original_device)

    return mask_dict