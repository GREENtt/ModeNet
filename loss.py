from tkinter import Variable

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, weight=None, reduction='mean'):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network.
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        '''
        super(CrossEntropyLoss2d, self).__init__()
        # self.cri = nn.NLLLoss2d(weight, ignore_index=255)
        self.cri = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, outputs, targets):
        # n,c,h,w = outputs.size()
        # outputs = outputs.view(-1,c)
        # targets =targets.view(-1)
        # # print('out',outputs.size(),'target',targets.size())
        loss = self.cri(outputs, targets)
        # print(loss)
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        # target = flatten(target)
        target = target.view(output.size(0), -1)   ####GREENT

        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, logits=False,reduce=True):
        super(FocalLoss, self).__init__()
        # if alpha is None:
        #     # self.alpha = Variable(torch.ones(class_num, 1))
        #     # self.alpha[0] = 0.3
        #     self.alpha = Variable(torch.tensor([0.1, 2, 1]))
        # else:
        #     if isinstance(alpha, Variable):
        #         self.alpha = alpha
        #     else:
        #         self.alpha = Variable(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs,targets,reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets,reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha*(1-pt)**self.gamma*BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# class focal_loss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
#         super(focal_loss, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, list):
#             assert len(alpha) == num_classes
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha < 1
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)
#
#         self.gamma = gamma
#
#     def forward(self, preds, labels):
#         # assert preds.dim()==2 and labels.dim()==1
#         preds = preds.view(-1, preds.size(-1))
#         self.alpha = self.alpha.to(preds.device)
#         preds_softmax = F.softmax(preds, dim=1)
#         preds_logsoft = torch.log(preds_softmax)
#
#         # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
#         preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
#         preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
#         self.alpha = self.alpha.gather(0, labels.view(-1))
#         # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
#         loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
#
#         loss = torch.mul(self.alpha, loss.t())
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss


class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)
        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)


# class focal_loss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
#
#         super(focal_loss, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, (float, int)):    #仅仅设置第一类别的权重
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)
#         if isinstance(alpha, list):  #全部权重自己设置
#             self.alpha = torch.Tensor(alpha)
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         alpha = self.alpha
#         print('aaaaaaa',alpha)
#         N = inputs.size(0)
#         C = inputs.size(1)
#         P = F.softmax(inputs,dim=1)
#         print('ppppppppppppppppppppp', P.size)
#         # ---------one hot start--------------#
#         class_mask = inputs.data.new(N, C).fill_(0)  # 生成和input一样shape的tensor
#         class_mask = class_mask.requires_grad_()  # 需要更新， 所以加入梯度计算
#         ids = targets.view(-1, 1)  # 取得目标的索引
#         print('取得targets的索引\n', ids.size)
#         class_mask.data.scatter_(1, ids.data, 1.)  # 利用scatter将索引丢给mask
#         print('targets的one_hot形式\n', class_mask)  # one-hot target生成
#         # ---------one hot end-------------------#
#         probs = (P * class_mask).sum(1).view(-1, 1)
#         print('留下targets的概率（1的部分），0的部分消除\n', probs)
#         # 将softmax * one_hot 格式，0的部分被消除 留下1的概率， shape = (5, 1), 5就是每个target的概率
#
#         log_p = probs.log()
#         print('取得对数\n', log_p)
#         # 取得对数
#         loss = torch.pow((1 - probs), self.gamma) * log_p
#         batch_loss = -alpha *loss.t()  # 對應下面公式
#         print('每一个batch的loss\n', batch_loss)
#         # batch_loss就是取每一个batch的loss值
#
#         # 最终将每一个batch的loss加总后平均
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         print('loss值为\n', loss)
#         return loss

################################################
# class focal_loss(nn.Module):
#     def __init__(self, alpha=[0.25, 0.9, 0.9], gamma=2, reduction='mean'):
#         """
#         :param alpha: 权重系数列表，三分类中第0类权重0.1，第1类权重10，第2类权重10
#         :param gamma: 困难样本挖掘的gamma
#         :param reduction:
#         """
#         super(focal_loss, self).__init__()
#         self.alpha = torch.tensor(alpha)
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         # print(target.size(),input.size(),self.alpha.size())
#         targets = target.data.new(target.size(0)).fill_(0)  ## 4
#         output = input.data.new(input.size(0), input.size(1)).fill_(0)  # 生成和input一样shape的tensor
#         alpha = self.alpha[targets].cuda() # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
#         output = output.requires_grad_()  # 需要更新， 所以加入梯度计算  torch.Size([4, 3])
#         log_softmax = F.log_softmax(output, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
#         logpt = torch.gather(log_softmax, dim=1, index=targets.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
#         logpt = logpt.view(-1)  # 降维，shape=(bs)
#         ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
#         pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
#         focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
#         if self.reduction == "mean":
#             return torch.mean(focal_loss)
#         if self.reduction == "sum":
#             return torch.sum(focal_loss)
#         return focal_loss
####################################

class focal_loss(nn.Module):
    def __init__(self, alpha=[0.1]+[0.45]*2, gamma=2, class_num=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, (float, int)):    #仅仅设置第一类别的权重
            self.alpha = torch.zeros(class_num)
            self.alpha = alpha
            self.alpha[1:] += (1 - alpha)
        if isinstance(alpha, list):  #全部权重自己设置
            self.alpha = torch.Tensor(alpha)
        self.gamma = gamma

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)  # 这里转成log(pt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        # print(input.size(),target.size(),logpt.size())
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# class focal_loss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
#         super(focal_loss, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, list):
#             assert len(alpha) == num_classes
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha < 1
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)
#         self.gamma = gamma
#
#     def forward(self, output, target):
#         # assert preds.dim()==2 and labels.dim()==1
#         output = F.softmax(output, dim=1)
#         output = flatten(output)
#         target = flatten(target)
#         output = output.view(-1, output.size(-1))
#         self.alpha = self.alpha.to(output.device)
#         preds_softmax = F.softmax(output, dim=1)
#         preds_logsoft = torch.log(preds_softmax)
#
#         # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
#         preds_softmax = preds_softmax.gather(1, target.view(-1, 1))
#         preds_logsoft = preds_logsoft.gather(1, target.view(-1, 1))
#         self.alpha = self.alpha.gather(0, target.view(-1))
#         # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
#         loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
#
#         loss = torch.mul(self.alpha, loss.t())
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=1, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.size_average = size_average
#         self.elipson = 0.000001
#
#     def forward(self, logits, labels):
#         """
#         cal culates loss
#         logits: batch_size * labels_length * seq_length
#         labels: batch_size * seq_length
#         """
#         if labels.dim() > 2:
#             labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
#             labels = labels.transpose(1, 2)
#             labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
#         if logits.dim() > 3:
#             logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
#             logits = logits.transpose(2, 3)
#             logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
#         # assert (logits.size(0) == labels.size(0))
#         # assert (logits.size(2) == labels.size(1))
#         batch_size = logits.size(0)
#         labels_length = logits.size(1)
#         seq_length = logits.size(2)
#
#         # transpose labels into labels onehot
#         new_label = labels.unsqueeze(1)
#         label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
#
#         # calculate log
#         log_p = F.log_softmax(logits)
#         pt = label_onehot * log_p
#         sub_pt = 1 - pt
#         fl = -self.alpha * (sub_pt) ** self.gamma * log_p
#         if self.size_average:
#             return fl.mean()
#         else:
#             return fl.sum()


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, output, target):

        if output.dim() > 2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss