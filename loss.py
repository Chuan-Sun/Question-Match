import math
import numpy as np
import random

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp


class RDropLoss(nn.Layer):
    def __init__(self, reduction='none'):
        super(RDropLoss, self).__init__()
        assert reduction in {'sum', 'mean', 'none', 'batchmean'}
        self.reduction = reduction

    def forward(self, p, q):
        p_loss = F.kl_div(F.log_softmax(p, axis=-1), F.softmax(q, axis=-1), reduction=self.reduction)
        q_loss = F.kl_div(F.log_softmax(q, axis=-1), F.softmax(p, axis=-1), reduction=self.reduction)

        loss = (p_loss + q_loss) / 2
        loss = loss.sum(1, keepdim=True)
        return loss


class CELoss(nn.Layer):
    def __init__(self, args):
        super(CELoss, self).__init__()
        self.LSR_coef   = args.LSR_coef
        self.hinge_coef = args.hinge_coef
        self.focal_coef = args.focal_coef

        self.ce = paddle.nn.loss.CrossEntropyLoss(
            reduction='none',
            soft_label=True if self.LSR_coef > 0 else False
        )

    def forward(self, logit, labels):
        # 标签平滑
        if self.LSR_coef > 0: labels = self.label_smooth(labels)

        ce_loss = self.ce(logit, labels)

        return ce_loss

    def label_smooth(self, labels):
        labels = F.one_hot(labels.squeeze(1), num_classes=2)
        labels = F.label_smooth(labels, epsilon=self.LSR_coef)
        return labels


class Loss(nn.Layer):
    def __init__(self, args, num_training_steps):
        super().__init__()
        self.rdrop_coef         = args.rdrop_coef
        self.rdrop_buffer       = args.rdrop_buffer
        self.self_dist          = args.self_dist
        self.contrastive_coef   = args.contrastive_coef
        self.num_training_steps = num_training_steps

        self.ce = CELoss(args)
        self.rdrop = RDropLoss()

    def forward(self, logit1, logit2, feats, labels, dist_logit, global_step, weight):
        ce_loss = self.ce(logit1, labels)

        # Double CE
        if logit2 is not None:
            ce_loss = (ce_loss + self.ce(logit2, labels)) / 2

        loss = ce_loss

        ramp_up_coef = self.ramp_up(global_step) if self.rdrop_buffer else 1

        # KL 散度
        kl_loss = 0
        if logit2 is not None and self.rdrop_coef > 0:
            kl_loss = self.rdrop(logit1, logit2)
            loss = loss + ramp_up_coef * self.rdrop_coef * kl_loss

        loss = (loss * weight).mean()
        ce_loss = (ce_loss * weight).mean()
        kl_loss = (kl_loss * weight).mean()

        return loss, ce_loss, kl_loss

    def ramp_up(self, step):
        if step < (self.num_training_steps / 3):
            p = step / (self.num_training_steps / 3)
            p = 1 - p
            return math.exp(-5 * p * p)
        else:
            return 1
