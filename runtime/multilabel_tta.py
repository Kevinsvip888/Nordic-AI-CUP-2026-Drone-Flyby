"""Optional class-score alternatives before NMS; no calibrated-probability claim."""
from types import MethodType
import torch
from ensemble_tta import EnsembleTTA

def top_two_scores(prediction,nc):
    value=prediction.clone()
    scores=value[:,4:4+nc]
    indices=scores.topk(min(2,nc),dim=1).indices
    keep=torch.zeros_like(scores,dtype=torch.bool).scatter_(1,indices,True)
    scores.masked_fill_(~keep,0.)
    return value

class MultiLabelTTA(EnsembleTTA):
    def __init__(self,base,mode='top2'):
        from ultralytics.utils import nms,ops
        assert mode in ('top2','all')
        predictor=base.model.predictor
        assert not predictor.model.end2end,'Raw one-to-many head required'
        self.original_postprocess=predictor.postprocess
        self.multilabel_mode=mode;self.raw_shapes=[];self.output_counts=[]
        owner=self
        def postprocess(predictor,preds,img,orig_imgs,**kwargs):
            assert not kwargs and getattr(predictor,'_feats',None) is None
            value=preds[0] if isinstance(preds,(tuple,list)) else preds
            nc=len(predictor.model.names)
            assert value.ndim==3 and value.shape[1]==nc+4
            owner.raw_shapes.append(list(value.shape))
            value=top_two_scores(value,nc) if mode=='top2' else value.clone()
            outputs=nms.non_max_suppression(value,predictor.args.conf,predictor.args.iou,
                classes=prediction_classes(predictor),agnostic=False,multi_label=True,
                max_det=predictor.args.max_det,nc=nc,end2end=False,rotated=False)
            owner.output_counts.extend(len(v) for v in outputs)
            if not isinstance(orig_imgs,list):orig_imgs=ops.convert_torch2numpy_batch(orig_imgs)[...,::-1]
            return predictor.construct_results(outputs,img,orig_imgs)
        predictor.postprocess=MethodType(postprocess,predictor)
        super().__init__([base],('original','h','v','hv'),False)

    def remove(self):self.bases[0].model.predictor.postprocess=self.original_postprocess

def prediction_classes(predictor):
    assert predictor.args.classes is None,'No class filtering is permitted in this experiment'
    return None
