"""
preds.shape: (N,1,H,W) || (N,1,H,W,D) || (N,H,W) || (N,H,W,D)
labels.shape: (N,1,H,W) || (N,1,H,W,D) || (N,H,W) || (N,H,W,D)
"""
import torch

class Dice:

    def __init__(self, name='Dice', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        # 检查输入是否包含NaN
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        
        res = {}
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            preds_ = (preds == class_index).to(torch.int)
            labels_ = (labels == class_index).to(torch.int)
            intersection = (preds_ * labels_).sum().float()
            union_sum = preds_.sum().float() + labels_.sum().float()
            
            if union_sum == 0:
                # 如果预测和标签都为空,Dice为1(完全匹配)
                res[class_name] = 1.0
            else:
                dice_value = (2.0 * intersection / union_sum)
                # 检查是否为NaN或Inf
                if torch.isnan(dice_value) or torch.isinf(dice_value):
                    res[class_name] = 0.0
                else:
                    res[class_name] = dice_value.item()
        return res


class Jaccard:
    def __init__(self, name='Jaccard', class_indexs=[1], class_names=['xx']) -> None:
        super().__init__()
        self.name = name
        self.class_indexs = class_indexs
        self.class_names = class_names

    def __call__(self, preds, labels):
        # 检查输入是否包含NaN
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        
        res = {}
        for class_index, class_name in zip(self.class_indexs, self.class_names):
            preds_ = (preds == class_index).to(torch.int)
            labels_ = (labels == class_index).to(torch.int)
            intersection = (preds_ * labels_).sum().float()
            union = ((preds_ + labels_) != 0).sum().float()
            
            if union == 0:
                # 如果预测和标签都为空,Jaccard为1(完全匹配)
                res[class_name] = 1.0
            else:
                jaccard_value = (intersection / union)
                # 检查是否为NaN或Inf
                if torch.isnan(jaccard_value) or torch.isinf(jaccard_value):
                    res[class_name] = 0.0
                else:
                    res[class_name] = jaccard_value.item()
        return res

