from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, jaccard_score

# metric
class F1_Score(nn.Module):
    def __init__(self, mode: str = 'multiclass', average: str = 'micro', from_logits: bool = True, threshold=0.5,
                 ignore_index: int | list = None):

        super().__init__()

        assert mode in {'binary', 'multiclass', 'multilabel'}  # BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
        self.mode = mode

        assert average in ['micro', 'macro', 'samples', 'weighted', 'binary', None]

        # ignore_index : list
        if ignore_index is None:
            ignore_index = []
        elif isinstance(ignore_index, int):
            ignore_index = [ignore_index]

        self.average = average
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):

        bs = y_true.size(0)  # batch size
        num_classes = y_pred.size(1)  # classes

        if self.from_logits:
            if self.mode == 'multiclass':  # MULTICLASS_MODE -> argmax
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            else:  # BINARY_MODE / MULTILABEL_MODE -> sigmoid & binarize
                y_pred = F.logsigmoid(y_pred).exp()
                y_pred = (y_pred > self.threshold).float()

        # check shape
        assert y_true.view(-1).shape == y_pred.view(-1).shape

        # y_pred, y_true flatten
        y_pred = y_pred.cpu().view(-1)
        y_true = y_true.cpu().view(-1)

        if self.mode == 'binary':
            assert num_classes == 1, "num_classes > 1 is not BINARY_MODE"
            num_classes += 1  # 1-channel predict but still a 2-classes [0, 1] problem

        # classes
        classes = np.arange(num_classes)

        # ignore_index
        if self.ignore_index:
            # classes should have included ignore_index
            assert all(index in classes for index in self.ignore_index), \
                f'classes (indexes) should contain ignore_index'
            classes = classes[np.setxor1d(classes, self.ignore_index)]

        # print(f'not ignore:{f1_score(y_true, y_pred, average=None, zero_division=0)}')
        # print(f'ignore:{f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)}')
        score = f1_score(y_true, y_pred, labels=classes, average=self.average, zero_division=0)

        return score

class IoU_Score(nn.Module):
    def __init__(self, mode: str = 'multiclass', average: str = 'micro', from_logits: bool = True, threshold=0.5,
                 ignore_index: int | list = None):

        super().__init__()

        assert mode in {'binary', 'multiclass', 'multilabel'}  # BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
        self.mode = mode

        assert average in ['micro', 'macro', 'samples', 'weighted', 'binary']

        # ignore_index : list
        if ignore_index is None:
            ignore_index = []
        elif isinstance(ignore_index, int):
            ignore_index = [ignore_index]

        self.average = average
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):

        bs = y_true.size(0)  # batch size
        num_classes = y_pred.size(1) # classes

        if self.from_logits:
            if self.mode == 'multiclass':  # MULTICLASS_MODE -> argmax
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            else:  # BINARY_MODE / MULTILABEL_MODE -> sigmoid & binarize
                y_pred = F.logsigmoid(y_pred).exp()
                y_pred = (y_pred > self.threshold).float()

        # check shape
        assert y_true.view(-1).shape == y_pred.view(-1).shape

        # y_pred, y_true flatten
        y_pred = y_pred.cpu().view(-1)
        y_true = y_true.cpu().view(-1)

        if self.mode == 'binary':
            assert num_classes == 1, "num_classes > 1 is not BINARY_MODE"
            num_classes += 1 # 1-channel predict but still a 2-classes [0, 1] problem

        # classes
        classes = np.arange(num_classes)

        # ignore_index
        if self.ignore_index:
            # classes should have included ignore_index
            assert all(index in classes for index in self.ignore_index), \
                f'classes (indexes) should contain ignore_index'
            classes = classes[np.setxor1d(classes, self.ignore_index)]

        # print(f'not ignore:{f1_score(y_true, y_pred, average=None, zero_division=0)}')
        # print(f'ignore:{f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)}')
        score = jaccard_score(y_true, y_pred, labels=classes, average=self.average, zero_division=0)

        return score

if __name__ == '__main__':

    B = 2
    C = 5
    H = W = 5
    size_pred = (B, 5, H, W)
    # size_true = (B, 1, H, W)
    size_true = (B, 1, H, W)
    # m = CustomizeLoss(loss='diouce', ).cuda()
    m = F1_Score(ignore_index=4, average=None).cuda()
    for i in range(100):
        pred = torch.rand(size_pred).cuda()
        true = torch.randint(5, size_true).cuda()  # multi_class

        y = m(pred, true)
        print(y)
