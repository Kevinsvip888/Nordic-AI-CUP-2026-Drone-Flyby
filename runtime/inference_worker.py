"""Keep CUDA initialization and inference on one persistent worker thread."""
from concurrent.futures import ThreadPoolExecutor

class InferenceWorker:
    def __init__(self):self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='drone-infer')
    def run(self,fn,*args,**kwargs):return self.executor.submit(fn,*args,**kwargs).result()
    def wrap(self,detector):
        worker=self
        class Proxy:
            def __call__(self,request):return worker.run(detector,request)
        return Proxy()
    def close(self):self.executor.shutdown(wait=True)
