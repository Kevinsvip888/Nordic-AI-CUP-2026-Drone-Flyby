"""Measured multi-view/model candidates using only the current received view."""
import numpy as np
from baseline_endpoint import INFERENCE_CONFIG
from detector_adapter import response_from_ultralytics
from dtos import DroneFlybyPredictionDto, DroneFlybyPredictResponseDto
from utils import decode_view
from input_tta import undo_flip, fuse

class EnsembleTTA:
    def __init__(self,bases,kinds=('original','h','v'),batch=False):
        self.bases=bases;self.kinds=kinds;self.batch=batch
        self.config={**INFERENCE_CONFIG,'imgsz':1280}
        for base in bases:
            warm=np.zeros((540,960,3),dtype=np.uint8)
            base.model.predict(source=[warm for _ in kinds] if batch else warm,**self.config)
            base._synchronize()

    def __call__(self,request):
        image=decode_view(request.view)
        sources=[image if k=='original' else np.ascontiguousarray(image[:,::-1] if k=='h' else image[::-1,:] if k=='v' else image[::-1,::-1]) for k in self.kinds]
        views=[]
        for base in self.bases:
            with base.lock:
                results=base.model.predict(source=sources,**self.config) if self.batch else [base.model.predict(source=s,**self.config)[0] for s in sources]
                for kind,result in zip(self.kinds,results):
                    response=response_from_ultralytics(request,result)
                    annotations=[]
                    for a in response.annotations:
                        item=a.model_dump()
                        cx=request.view.center_x/request.original_width;cy=request.view.center_y/request.original_height
                        box=undo_flip(item['bbox'],'h' if kind=='hv' else kind,cx,cy)
                        if kind=='hv':box=undo_flip(box,'v',cx,cy)
                        item['bbox']=[min(1.,max(0.,v)) for v in box];annotations.append(item)
                    views.append(annotations)
        return DroneFlybyPredictResponseDto(request_id=request.request_id,frame=request.frame,
            annotations=[DroneFlybyPredictionDto(**a) for a in fuse(views,'wbf')],requested_view=None)
