import torch

def delete_cm_class(class_: "list[int]", cm):
    # Swap row and col to last index
    new_indices = swap_indices(cm, class_, -1)
    cm = torch.index_select(cm, 0, new_indices)
    cm = torch.index_select(cm, 1, new_indices)
    # Delete last row and col
    cm = cm[:-1, :-1]
    return cm

def swap_indices(cm, index1, index2):
    nb_classes = cm.shape[0]
    indices = list(range(nb_classes))
    aux = indices[index1], indices[index2]
    indices[index2], indices[index1] = aux
    return torch.LongTensor(indices)


def ovr_cm(target_class, cm):
     # Swap row and col to last index
    new_indices = swap_indices(cm, target_class, -1)
    cm = torch.index_select(cm, 0, new_indices)
    cm = torch.index_select(cm, 1, new_indices)
    cm00 = cm[:-1, :-1].sum()
    cm01 = cm[:-1, -1].sum()
    cm10 = cm[-1, :-1].sum()
    cm11 = cm[-1, -1]
    return torch.Tensor([[cm00, cm01], [cm10, cm11]]).type(torch.DoubleTensor)

def get_cm(prediction, target):
    nb_classes = prediction.shape[1]
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    _, preds = torch.max(prediction, 1)
    for t, p in zip(target.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

def DiceCoefficient(prediction, target, ignore_index=None, only_consider=None):
    cm = get_cm(prediction, target).type(torch.DoubleTensor)

    if ignore_index:
        cm = delete_cm_class(ignore_index, cm)
    if only_consider:
        cm = ovr_cm(only_consider, cm)

    dice = 2.0 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + 1e-15)
    return dice

def iou_pytorch(prediction: torch.Tensor, target: torch.Tensor, SMOOTH = 1e-6, ignore_index=None):
    _, preds = torch.max(prediction, 1)
    if ignore_index:
        filter_ = (preds != ignore_index) * (target != ignore_index)
        preds = preds[filter_]
        target = target[filter_]
        intersection = (preds & target).float().mean()
        union = (preds | target).float().mean()
    else:
        intersection = (preds & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (preds | target).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch