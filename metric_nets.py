"""
Copy & paste on the notebook, and run once.
! wget https://raw.githubusercontent.com/ronghuaiyang/arcface-pytorch/master/models/metrics.py
! wget https://raw.githubusercontent.com/KaiyangZhou/pytorch-center-loss/master/center_loss.py
"""
from utils import *
from metrics import *
from center_loss import *


class LabelCatcher(LearnerCallback):
    last_labels = None

    def __init__(self, learn:Learner):
        super().__init__(learn)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        LabelCatcher.last_labels = last_target
        return {'last_input': last_input, 'last_target': last_target} 


class MerginFC(nn.Module):
    def __init__(self, c, xface_product=ArcMarginProduct, m=0.5):
        super().__init__()
        self.metric_fc = xface_product(512, c, m=m)
    
    def forward(self, x):
        labels = LabelCatcher.last_labels if self.training else torch.tensor([0]*len(x)).cuda()
        x = self.metric_fc(x, labels)
        return x


def XFaceNet(org_model, data, xface_product=ArcMarginProduct, m=0.5):
    body, feat_head = body_feature_model(org_model)
    metric_fc = MerginFC(data.c, xface_product=xface_product, m=m).cuda()
    return nn.Sequential(body, nn.Sequential(feat_head, metric_fc))


class L2ConstrainedLayer(nn.Module):
    def __init__(self, alpha=16):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        l2 = torch.sqrt((x**2).sum())
        x = self.alpha * (x / l2)
        return x


def L2ConstrainedNet(org_model, alpha=16):
    body, feat_head = org_model.children()
    l2constrain = L2ConstrainedLayer(alpha=alpha).cuda()
    return nn.Sequential(body, nn.Sequential(feat_head, l2constrain))


class WrapCenterLoss(nn.Module):
    "CenterLoss wrapper for https://github.com/KaiyangZhou/pytorch-center-loss."
    
    def __init__(self, learn, data, weight_cent=1/10):
        super().__init__()
        self.org_loss = learn.loss_func
        self.center_loss = CenterLoss(data.c, data.c)
        self.weight_cent = weight_cent

    def forward(self, output, target):
        dL = self.org_loss(output, target)
        dC = self.center_loss(output, target)
        #print(dL, dC)
        d = dL + self.weight_cent * dC
        return d
