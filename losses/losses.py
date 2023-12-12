from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import BCEWithLogitsLoss, L1Loss
from einops import rearrange

# loss

# LL1Loss

# BCEWithLogitsLoss Pytorch
"""
Note: CARLoss now
doesn't use threshold
doesn't use no_grad on class center
"""

class CARLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 6,
                 threshold: float = 1e-10,
                 apply_filter: bool = False,
                 num_filter: int = 512,
                 window_size: int = 2,
                 ):

        super().__init__()

        self.num_classes = num_classes

        self.apply_fiter = apply_filter
        self.filter = nn.Conv2d(num_classes, num_filter, kernel_size=1)

        self.ws = window_size

        self.loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=-1)
        self.threshold = threshold
    def divide_no_nan(self,
                      x: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        """
        proximate the function like tensorflow `tf.math.divide_no_nan
        replace NaN or Inf to 0
        https://github.com/autonlab/nbeats/blob/master/nbeats/contrib/utils/pytorch/losses.py

        :param x:
        :param y:
        :return:
        """
        div = x / y
        div[div != div] = 0.  # handle NaN
        div[div == float('inf')] = 0.  # handle Inf

        return div

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_true (label) : [B, H, W]
        """

        # check batch size
        assert y_true.size(0) == y_pred.size(0)

        B = y_true.size(0)
        H = y_pred.size(2)
        W = y_pred.size(3)

        if self.apply_fiter:
            y_pred = self.filter(y_pred)

        C = y_pred.size(1)  # pred classes

        ##############################################################################
        # feat flatten
        y_pred = y_pred.view(B, C, -1)  # flatten

        # label one-hot & flatten
        y_true = F.one_hot(y_true.to(torch.long), self.num_classes).float()  # one-hot [B, H, W] -> [B, H, W, C]
        y_true = y_true.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        y_true = y_true.view(B, self.num_classes, -1)  # flatten [B, C, H, W] -> [B, C, (H * W)]
        ##############################################################################

        # dot product
        # window size
        ws = self.ws
        h = H // self.ws
        w = W // self.ws

        non_zero_map = torch.count_nonzero(y_true, dim=-1)
        feat = rearrange(y_pred, 'b c (h hsp w wsp) -> (b h w) (hsp wsp) c', h=h, w=w, hsp=ws, wsp=ws)
        label = rearrange(y_true, 'b c (h hsp w wsp) -> (b h w) (hsp wsp) c', h=h, w=w, hsp=ws, wsp=ws)

        label_scale = self.divide_no_nan(label, non_zero_map)
        # with torch.no_grad():
        class_center = label_scale.transpose(-2, -1) @ feat  # µ = (Y.t @ X) / Y.non_zero

        # intra-c2p # consider ignore label?
        intra_c2p_loss = self.loss(feat, label @ class_center)  # (1 - σ) |Y @ µ - X|
        print(f'{intra_c2p_loss=}')

        # intra-c2p
        # c2c = nn.Softmax() # add scale : "/ sqrt(C)"
        c2c = (class_center @ class_center.transpose(-2, -1)) * (self.num_classes ** -0.5)  # softmax((µ.t @ µ) / sqrt(C))
        d_c2c = (1 - torch.eye(self.num_classes)) * c2c  # (1 - eye(num_classes)) * c2c
        # d_c2c = (d_c2c - (self.threshold / (self.num_classes - 1))).clamp(min=0)  # - or 0 correlation is okay?
        c2c_loss = self.loss(d_c2c, torch.zeros_like(d_c2c))
        print(f'{c2c_loss=}')

        # c2p
        c2p = (1 - label) * (feat @ class_center.transpose(-2, -1))   # (1-Y) * µ.t @ X

        # diagonal (dim=1, dim2=2)
        # [B, num_classes]
        c_diag = torch.diagonal(class_center @ class_center.transpose(-2, -1), dim1=1, dim2=2)  # diag(µ.t @ µ)
        c_diag = c_diag.view(-1, 1, self.num_classes)

        c2p = c2p + c_diag
        c2p = self.softmax(c2p)
        # (1 -Y) * c2p
        # d_c2p - (threshold / (num_classes - 1))
        # (1 -Y) * c2p - (threshold / (num_classes - 1))
        # d_c2p = ((1 - label) * c2p - (self.threshold / (self.num_classes - 1))).clamp(min=0)
        d_c2p = (1 - label) * c2p
        c2p_target = (1 - label) * torch.ones_like(d_c2p) * (1 / self.num_classes)

        # c2p_loss
        inter_c2p_loss = self.loss(d_c2p, c2p_target)
        print(f'{inter_c2p_loss=}')

        loss = intra_c2p_loss + c2c_loss + inter_c2p_loss


        return loss


class CARLoss_intra_c2p(nn.Module):
    def __init__(self,
                 num_classes: int = 6,
                 threshold: float = 1e-10,
                 apply_filter: bool = False,
                 num_filter: int = 512,
                 window_size: int = 2,
                 ):
        super().__init__()

        self.num_classes = num_classes

        self.apply_fiter = apply_filter
        self.filter = nn.Conv2d(num_classes, num_filter, kernel_size=1)

        self.ws = window_size

        self.loss = nn.MSELoss()
        self.softmax = nn.Softmax(dim=-1)
        self.threshold = threshold

    def divide_no_nan(self,
                      x: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        """
        proximate the function like tensorflow `tf.math.divide_no_nan
        replace NaN or Inf to 0
        https://github.com/autonlab/nbeats/blob/master/nbeats/contrib/utils/pytorch/losses.py

        :param x:
        :param y:
        :return:
        """
        div = x / y
        div[div != div] = 0.  # handle NaN
        div[div == float('inf')] = 0.  # handle Inf

        return div

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_true (label) : [B, H, W]
        """

        # check batch size
        assert y_true.size(0) == y_pred.size(0)

        B = y_true.size(0)
        H = y_pred.size(2)
        W = y_pred.size(3)

        if self.apply_fiter:
            y_pred = self.filter(y_pred)

        C = y_pred.size(1)  # pred classes

        ##############################################################################
        # feat flatten
        y_pred = y_pred.view(B, C, -1)  # flatten

        # label one-hot & flatten
        y_true = F.one_hot(y_true.to(torch.long), self.num_classes).float()  # one-hot [B, H, W] -> [B, H, W, C]
        y_true = y_true.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        y_true = y_true.view(B, self.num_classes, -1)  # flatten [B, C, H, W] -> [B, C, (H * W)]
        ##############################################################################

        # dot product
        # window size
        ws = self.ws
        h = H // self.ws
        w = W // self.ws

        feat = rearrange(y_pred, 'b c (h hsp w wsp) -> (b h w) (hsp wsp) c', h=h, w=w, hsp=ws, wsp=ws)
        label = rearrange(y_true, 'b c (h hsp w wsp) -> (b h w) (hsp wsp) c', h=h, w=w, hsp=ws, wsp=ws)

        non_zero_map = torch.count_nonzero(y_true, dim=-1).unsqueeze(-1)
        # non_zero_map = torch.sum((y_true != 0), dim=-1, keepdim=True)
        # print(non_zero_map.shape)
        y_true_scale = self.divide_no_nan(y_true, non_zero_map)
        label_scale = rearrange(y_true_scale, 'b c (h hsp w wsp) -> (b h w) (hsp wsp) c', h=h, w=w, hsp=ws, wsp=ws)

        # label_scale = self.divide_no_nan(label, non_zero_map)
        # with torch.no_grad():
        class_center = label_scale.transpose(-2, -1) @ feat  # µ = (Y.t @ X) / Y.non_zero

        # intra-c2p # consider ignore label?
        intra_c2p_loss = self.loss(feat, label @ class_center)  # (1 - σ) |Y @ µ - X|

        loss = intra_c2p_loss

        return loss

class DiceLoss(nn.Module):

    def __init__(self, mode: str = 'multiclass', from_logits: bool = True, weights: list = None, eps=1e-7,
                 ignore_index: int | list[int] = None, smooth: float = 0.1):
        super().__init__()

        assert mode in {'binary', 'multiclass', 'multilabel'}  # BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
        self.mode = mode


        # No ignore_index in BINARY_MODE
        if ignore_index is None or self.mode == 'binary':  # BINARY_MODE
            ignore_index = []
        elif isinstance(ignore_index, int):
            ignore_index = [ignore_index]

        # weights
        # No weight factor in BINARY_MODE
        if weights is None or self.mode == 'binary':  # BINARY_MODE
            weights = torch.Tensor([])
        else:
            weights = torch.Tensor(weights)  # to tensor
            weights /= weights.sum()  # normalize

        self.weights = weights
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # check batch size
        assert y_true.size(0) == y_pred.size(0)

        bs = y_true.size(0)  # batch size
        pred_classes = y_pred.size(1)  # pred classes
        hw = y_pred.size(2) * y_pred.size(3)  # h*w

        # num_classes > 1 or true_classes > 1 is not BINARY_MODE
        if self.mode == 'binary':
            assert pred_classes == 1, "pred_classes > 1 is not BINARY_MODE"

        num_classes = pred_classes

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result
            if self.mode == 'multiclass':  # MULTICLASS_MODE -> softmax
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:  # BINARY_MODE / MULTILABEL_MODE -> sigmoid
                y_pred = F.logsigmoid(y_pred).exp()

        dims = (0, 2)

        if self.mode == 'binary':  # BINARY_MODE
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            # no ignore_index in BINARY_MODE

        if self.mode == 'multiclass':  # MULTICLASS_MODE
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, pred_classes, -1)  # B, C, H*W

            if self.ignore_index:
                mask = torch.ones_like(y_true)  # create blank mask
                for index in self.ignore_index:  # iterate through ignore_index
                    mask *= y_true != index
                y_true = y_true * mask

            y_true = F.one_hot(y_true.to(torch.long), num_classes)  # B, H*W -> B, H*W, C
            y_true = y_true.permute(0, 2, 1)  # B, C, H*W

        if self.mode == 'multilabel':  # MULTILABEL_MODE
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index:
                mask = torch.ones_like(y_true)  # create blank mask
                for index in self.ignore_index:  # iterate through ignore_index
                    mask *= y_true != index
                y_pred = y_pred * mask
                y_true = y_true * mask

        # dice
        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        loss = 1 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)
        # print(len(loss))

        # weights
        if self.weights.numel():
            assert len(loss) == len(self.weights), f'weights have to be same length (size) of loss'
            return self.weighted_loss(loss, self.weights)
        else:
            return self.aggregate_loss(loss)

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return self.soft_dice_score(output, target, smooth, eps, dims)

    def soft_dice_score(self, output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7,
                        dims=None, ) -> torch.Tensor:
        assert output.size() == target.size()

        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)

        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

        return dice_score

    def aggregate_loss(self, loss):
        return loss.mean()

    def weighted_loss(self, loss, weights):
        # to device
        loss *= weights.to(loss.device)
        return loss.sum()

class CELoss(nn.Module):
    def __init__(self, mode: str = 'multiclass', from_logits: bool = True, weight: list = None,
                 ignore_index: int | list[int] = None,
                 smooth: float = 0.1):
        super().__init__()

        assert mode in {'binary', 'multiclass', 'multilabel'}

        self.from_logits = from_logits
        self.weight = weight
        self.ignore_index = ignore_index if ignore_index is not None else -100
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_true (label) : [B, H, W]
        """
        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=1).exp()

        if self.weight:
            assert len(self.weight) == y_pred.size(1), \
                f' a manual rescaling weight given to each class. If given, has to be size of C'

            # to tensor
            self.weight = torch.Tensor(self.weight).to(y_pred.device)

        ce_loss = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, label_smoothing=self.smooth)
        loss = ce_loss(y_pred, y_true)

        return loss


"""
[PRLoss] CustomizeLoss(loss='pr')
[DiouceLoss] CustomizeLoss(loss='diouce')

*PR = Precision & Recall
*Diouce = Dice & Iou 
"""

class CustomizeLoss(nn.Module):
    def __init__(self, loss: str, mode: str = 'multiclass', from_logits: bool = True, weights: list = None, eps=1e-7,
                 ignore_index: int | list = None, threshold=0.5):
        super().__init__()

        assert loss in {'pr', 'diouce'}
        self.loss = loss

        assert mode in {'binary', 'multiclass', 'multilabel'}  # BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
        self.mode = mode

        # ignore_index : list
        if ignore_index is None:
            ignore_index = []
        elif isinstance(ignore_index, int):
            ignore_index = [ignore_index]

        # weights
        # No weight factor in BINARY_MODE
        if weights is None or self.mode == 'binary':  # BINARY_MODE
            weights = torch.Tensor([])  # to tensor
        else:
            weights = torch.Tensor(weights)  # to tensor
            weights /= weights.sum()  # normalize

        self.weights = weights
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.eps = eps
        self.threshold = threshold

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # check batch size
        assert y_true.size(0) == y_pred.size(0)

        bs = y_true.size(0)  # batch size
        num_classes = y_pred.size(1)  # classes

        if self.mode == 'binary':
            assert num_classes == 1, "num_classes > 1 is not BINARY_MODE"
            num_classes += 1  # 1-channel predict but still a 2-classes [0, 1] problem

        # labels (classes) [0 ~ classes - 1]
        classes = np.arange(num_classes)

        if self.from_logits:
            if self.mode == 'multiclass':  # MULTICLASS_MODE -> argmax
                y_pred = torch.argmax(y_pred, dim=1)
            else:  # BINARY_MODE / MULTILABEL_MODE -> sigmoid & binarize
                y_pred = F.logsigmoid(y_pred).exp()
                y_pred = (y_pred > self.threshold).float()

        # flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
        dice_list = []
        diouce_list = []

        # ignore_index
        if self.ignore_index:
            # classes should contains ignore_index
            assert all(index in classes for index in self.ignore_index), \
                f'classes (indexes) should contain ignore_index'
            classes = classes[np.setxor1d(classes, self.ignore_index)]

        for class_id in classes:
            # separate the prediction and ground truth by labels
            # True, False (eq) -> 1, 0 (float)
            y_true = torch.eq(y_true, class_id).float()
            y_pred = torch.eq(y_pred, class_id).float()

            tp = torch.sum(y_true * y_pred)
            fp = torch.sum((1 - y_true) * y_pred)
            fn = torch.sum(y_true * (1 - y_pred))
            tn = torch.sum((1 - y_true) * (1 - y_pred))

            precision = tp / (tp + fp).clamp_min(self.eps)
            recall = tp / (tp + fn).clamp_min(self.eps)
            f1 = (precision * recall) / (precision + recall).clamp_min(self.eps)
            iou = tp / (tp + fp + fn).clamp_min(self.eps)
            dice = 2 * tp / (2 * tp + fp + fn).clamp_min(self.eps)
            diouce = iou * dice

            # list append
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            iou_list.append(iou)
            dice_list.append(dice)
            diouce_list.append(diouce)

            del precision, recall, f1, iou, dice, diouce

        # list to tensor & to device
        precision_list = torch.Tensor(precision_list).to(y_pred.device)
        recall_list = torch.Tensor(recall_list).to(y_pred.device)
        f1_list = torch.Tensor(f1_list).to(y_pred.device)
        iou_list = torch.Tensor(iou_list).to(y_pred.device)
        dice_list = torch.Tensor(dice_list).to(y_pred.device)
        diouce_list = torch.Tensor(diouce_list).to(y_pred.device)

        # weights is not None
        if self.weights.numel():
            assert len(self.weights) == len(classes), \
                f'weights have to be same length (size) of list'

            weighted_precision = (self.weights * precision_list).sum()
            weighted_recall = (self.weights * recall_list).sum()
            weighted_f1 = (self.weights * f1_list).sum()
            weighted_iou = (self.weights * iou_list).sum()
            weighted_dice = (self.weights * dice_list).sum()
            weighted_diouce = (self.weights * diouce_list).sum()

        else:
            weighted_precision = precision_list.mean()
            weighted_recall = recall_list.mean()
            weighted_f1 = f1_list.mean()
            weighted_iou = iou_list.mean()
            weighted_dice = dice_list.mean()
            weighted_diouce = diouce_list.mean()

        if self.loss == 'pr':
            return 2 - weighted_precision - weighted_recall
        if self.loss == 'diouce':
            return 1 - weighted_diouce

class DiceLoss_old(nn.Module):
    """
    Modified from
    qubvel /segmentation_models.pytorch
    https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/dice.py

    pred_less    --> apply to y_pred (change(expand) y_pred size)
    ignore_index --> apply to loss   (list)

    e.g. your model output predict `5` labels (y_pred), but there are `6` ground truth labels (y_true),
    in this situation you're pred_less

    --

    has    ignore_index & pred_less=True  : your model did not predict the ignore_index label
    has    ignore_index & pred_less=False : your model did     predict the ignore_index label but still ignored loss
    has no ignore_index & pred_less=True : Wrong


    multilabel_mode loss is acting weird (sometimes negative, overall value is too low)
    """

    def __init__(self, mode: str = 'multiclass', from_logits: bool = True, weights: list = None, eps=1e-7,
                 ignore_index: int | list[int] = None, smooth: float = 0.1, pred_less=False):
        super().__init__()

        assert mode in {'binary', 'multiclass', 'multilabel'}  # BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
        self.mode = mode

        # ignore_index : list
        assert (ignore_index is not None or self.mode == 'binary') and pred_less is True, \
            f'You must have ignore_label and not binary mode to predict less'  # BINARY_MODE

        # No ignore_index in BINARY_MODE
        if ignore_index is None or self.mode == 'binary':  # BINARY_MODE
            ignore_index = []
        elif isinstance(ignore_index, int):
            ignore_index = [ignore_index]

        # weights
        # No weight factor in BINARY_MODE
        if weights is None or self.mode == 'binary':  # BINARY_MODE
            weights = torch.Tensor([])
        else:
            weights = torch.Tensor(weights)  # to tensor
            weights /= weights.sum()  # normalize

        self.weights = weights
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.pred_less = pred_less

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # check batch size
        assert y_true.size(0) == y_pred.size(0)

        bs = y_true.size(0)  # batch size
        pred_classes = y_pred.size(1)  # pred classes
        hw = y_pred.size(2) * y_pred.size(3)  # h*w

        # num_classes > 1 or true_classes > 1 is not BINARY_MODE
        if self.mode == 'binary':
            assert pred_classes == 1, "pred_classes > 1 is not BINARY_MODE"

        if self.pred_less:  # num_classes = predicted classes + len of ignore_index
            num_classes = pred_classes + len(self.ignore_index)
        else:
            num_classes = pred_classes

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result
            if self.mode == 'multiclass':  # MULTICLASS_MODE -> softmax
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:  # BINARY_MODE / MULTILABEL_MODE -> sigmoid
                y_pred = F.logsigmoid(y_pred).exp()

        dims = (0, 2)

        if self.mode == 'binary':  # BINARY_MODE
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == 'multiclass':  # MULTICLASS_MODE
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, pred_classes, -1)  # B,H*W, C

            y_true = F.one_hot(y_true.to(torch.long), num_classes)  # B,H*W -> B,H*W, C
            y_true = y_true.permute(0, 2, 1)  # B, C, H*W

            if self.pred_less:
                # create a zeros tensor (y_pred_expand) size like y_true & permute to [C, B, H*W] (from [B, C, H*W])
                y_pred_expand = torch.zeros_like(y_true).permute(1, 0, 2)
                # also permute y_pred to [C, B, H*W] (from [B, C, H*W]) & index append(?) y_pred to y_pred_expand
                y_pred_expand[0:pred_classes] = y_pred.clone().permute(1, 0, 2)
                # finally, permute back to [B, C, H*W]
                y_pred_expand = y_pred_expand.permute(1, 0, 2)  # [C, B, H*W] -> [B, C, H*W]
                # y_pred_expand is the new y_pred
                y_pred = y_pred_expand

        if self.mode == 'multilabel':  # MULTILABEL_MODE
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        # dice
        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        loss = 1 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        # ignore_index
        if self.ignore_index:
            classes = np.arange(num_classes)
            classes = classes[np.setxor1d(classes, self.ignore_index)]
            loss = loss[classes]

        # weights
        if self.weights.numel():
            assert len(loss) == len(self.weights), f'weights have to be same length (size) of loss'
            return self.weighted_loss(loss, self.weights)
        else:
            return self.aggregate_loss(loss)

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return self.soft_dice_score(output, target, smooth, eps, dims)

    def soft_dice_score(self, output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7,
                        dims=None, ) -> torch.Tensor:
        assert output.size() == target.size()

        if dims is not None:
            intersection = torch.sum(output * target, dim=dims)
            cardinality = torch.sum(output + target, dim=dims)
        else:
            intersection = torch.sum(output * target)
            cardinality = torch.sum(output + target)

        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

        return dice_score

    def aggregate_loss(self, loss):
        return loss.mean()

    def weighted_loss(self, loss, weights):
        # to device
        loss *= weights.to(loss.device)
        return loss.sum()

if __name__ == '__main__':

    B = 2
    C = 5
    H = W = 5
    size_pred = (B, 5, H, W)
    # size_true = (B, 1, H, W)
    size_true = (B, 1, H, W)
    # m = CustomizeLoss(loss='diouce', ).cuda()
    m = DiceLoss(ignore_index=5,).cuda()
    for i in range(100):
        pred = torch.rand(size_pred).cuda()
        true = torch.randint(5, size_true).cuda()  # multi_class

        y = m(pred, true)
        print(y)
