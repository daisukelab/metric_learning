from dlcliche.fastai import *

try:
    from metrics import *
except:
    ! wget https://raw.githubusercontent.com/ronghuaiyang/arcface-pytorch/master/models/metrics.py
    from metrics import *

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
