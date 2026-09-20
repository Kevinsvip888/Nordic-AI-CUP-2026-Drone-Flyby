"""Frozen 1280 FP32 three-view TTA. Only transforms the received image."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from baseline_endpoint import INFERENCE_CONFIG
from detector_adapter import response_from_ultralytics
from dtos import DroneFlybyPredictionDto,DroneFlybyPredictResponseDto
from utils import decode_view
from temporal_camera import iou

def undo_flip(box,kind,center_x=.5,center_y=.5):
    x1,y1,x2,y2=box
    if kind=='h':return [2*center_x-x2,y1,2*center_x-x1,y2]
    if kind=='v':return [x1,2*center_y-y2,x2,2*center_y-y1]
    return list(box)

def fuse(views,mode):
    entries=sorted([(a,k) for k,v in enumerate(views) for a in v],key=lambda x:-x[0]['confidence'])
    clusters=[]
    for a,k in entries:
        choices=[(iou(a['bbox'],c['bbox']),j) for j,c in enumerate(clusters) if c['label']==a['object_id'] and (mode=='nms' or k not in c['views'])]
        match=max(choices,default=(0,None))
        if match[0]>=.55:
            if mode=='nms':continue
            c=clusters[match[1]];weight=a['confidence'];total=c['weight']+weight
            c['bbox']=[(x*c['weight']+y*weight)/total for x,y in zip(c['bbox'],a['bbox'])]
            c['weight']=total;c['views'].add(k)
        else:clusters.append({'label':a['object_id'],'bbox':list(a['bbox']),'weight':a['confidence'],'views':{k}})
    output=[{'object_id':c['label'],'bbox':c['bbox'],'confidence':c['weight']/len(views) if mode=='wbf' else c['weight']} for c in clusters]
    return sorted(output,key=lambda a:-a['confidence'])[:300]

class TTADetector:
    def __init__(self,base,mode,imgsz=1280):
        assert mode in ('nms','wbf')
        self.base=base;self.mode=mode;self.shapes=[];self.config={**INFERENCE_CONFIG,'imgsz':imgsz}
        original=base.model.predictor.preprocess
        def preprocess(im):
            t=original(im);self.shapes.append(list(t.shape));return t
        base.model.predictor.preprocess=preprocess
        self.hook=SimpleNamespace(remove=lambda:setattr(base.model.predictor,'preprocess',original))
        base.model.predict(source=np.zeros((540,960,3),dtype=np.uint8),**self.config);base._synchronize()

    def __call__(self,request):
        with self.base.lock:
            image=decode_view(request.view);views=[]
            for kind in ('original','h','v'):
                source=image if kind=='original' else np.ascontiguousarray(image[:,::-1] if kind=='h' else image[::-1,:])
                result=self.base.model.predict(source=source,**self.config)[0]
                response=response_from_ultralytics(request,result)
                annotations=[]
                for a in response.annotations:
                    item=a.model_dump();item['bbox']=undo_flip(item['bbox'],kind,request.view.center_x/request.original_width,request.view.center_y/request.original_height)
                    item['bbox']=[min(1.,max(0.,v)) for v in item['bbox']];annotations.append(item)
                views.append(annotations)
            output=fuse(views,self.mode)
            return DroneFlybyPredictResponseDto(request_id=request.request_id,frame=request.frame,
                annotations=[DroneFlybyPredictionDto(**a) for a in output],requested_view=None)
