import torch
import inspect
from packaging import version
from torch.utils.data import DataLoader


def calculate_the_importance_label(model, task, data_loader, num_samples, cuda_device, grad_type):
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
    
    if task in ["vqa", "gqa"]:
        bce_loss = torch.nn.BCEWithLogitsLoss()
    else:
        bce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for idx, (ques_id, feats, boxes, sent, target) in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)
        feats, boxes, target = feats.to(cuda_device), boxes.to(cuda_device), target.to(cuda_device)
        logits = model(feats, boxes, sent)
        if task in ["vqa", "gqa"]:
            loss = bce_loss(logits, target)
        else:
            loss = bce_loss(logits, target.squeeze(-1))

        # inputs.pop("idx", None)
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         inputs[k] = v.to(cuda_device)

        # return_dicts = model(**inputs)

        # loss = return_dicts["loss"]

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients_dict[name] += grad_method(param.grad).data
        
        model.zero_grad()

    return gradients_dict


def calculate_the_importance_expect(model, task, data_loader, num_samples, cuda_device, grad_type):
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

    for idx, (ques_id, feats, boxes, sent, target) in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        # inputs.pop("idx", None)
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         inputs[k] = v.to(cuda_device)

        # return_dicts = model(**inputs)

        # logits = return_dicts["logits"]

        feats, boxes, target = feats.to(cuda_device), boxes.to(cuda_device), target.to(cuda_device)
        logits = model(feats, boxes, sent)

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


def compute_fisher(model, task, train_dataset, data_collator, num_samples):
    importance_method = calculate_the_importance_label

    # import pdb
    # pdb.set_trace()

    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True
    )
    
    grad_type = "square"

    return importance_method(model, task, data_loader, num_samples, cuda_device, grad_type)
