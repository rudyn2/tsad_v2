import torch.nn as nn
import torch
from losses import DiceLoss, WeightedPixelWiseNLLoss, FocalLoss
from torch import Tensor
from typing import Tuple, Dict


class ADLoss(nn.Module):
    def __init__(self,
                 loss_weights: Tuple,
                 seg_loss: str,
                 device: str = 'cuda'):
        """
        @param loss_weights: Loss weights associated to segmentation, traffic light, pedestrian and vehicle
                             affordances loss. In that order.
        @param tl_weights:  Weights associated to traffic light classification. Expected: (green_status_weight,
                            red_status_weight).
        @param seg_loss: Type of segmentation loss.
        @param device: Device used for loss calculation. Expected: cpu or cuda.


        """
        super(ADLoss, self).__init__()

        self._device = device
        if seg_loss == 'dice':
            self._seg_loss = DiceLoss()
        elif seg_loss == 'wnll':
            weights = {
                0: 25,
                1: 15,
                2: 10,
                3: 10,
                4: 10,
                5: 1,
                6: 25
            }
            sum_weights = sum(weights.values())
            # normalize weights
            for k, v in weights.items():
                weights[k] = v/sum_weights
            self._seg_loss = WeightedPixelWiseNLLoss(weights)
        else:
            self._seg_loss = FocalLoss()

        self._loss_weights = loss_weights

    def __call__(self, prediction: Tensor, target: Tensor) -> Dict[str, torch.FloatTensor]:
        """
        Calculates weighted loss.
        @param prediction: Dictionary with predictions. The following format is expected:
                            {
                                'segmentation': torch.Tensor(...),
                                'traffic_light_status': torch.Tensor(...),
                                'vehicle_affordances': torch.Tensor(...)
                            }
        @param target: Dictionary with expected values or ground truth. The following format is expected:
                            {
                                'segmentation': torch.Tensor(...),
                                'traffic_light_status': torch.Tensor(...),
                                'vehicle_affordances': torch.Tensor(...),
                            }
        @return: <class Dict> {
            'loss': torch.FloatTensor,
            'seg_loss': torch.FloatTensor,
            'tl_loss': torch.FloatTensor,
            'va_loss': torch.FloatTensor
        }
        """
        return self._seg_loss(prediction, target)
