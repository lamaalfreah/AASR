"""Lazy, thread-safe trained Short → calibrated Router → optional frozen Long."""
import threading
from functools import lru_cache
from django.conf import settings
from language.neural_short import NeuralShortParser
from language.router_v2 import RouterV2
from language.long_parser import LongParser,ProviderResponse
from spatial.schema import OPERATIONS
from .router_features import safe_features


class ModalProvider:
    def complete(self,messages):
        import modal
        from language.modal_checkpoint import digest,GENERATION,MODEL
        key=digest(dict(namespace='aasr-serving-v1',messages=messages,revision=settings.AASR_MODAL_REVISION,generation=GENERATION))
        remote=modal.Cls.from_name(settings.AASR_MODAL_APP,'QwenLong')(revision=settings.AASR_MODAL_REVISION)
        call=remote.generate.spawn(messages,key)
        try:result=call.get(timeout=settings.AASR_LONG_TIMEOUT)
        except TimeoutError:
            call.cancel()
            raise
        if result.get('request_digest')!=key or result.get('model')!=MODEL or result.get('revision')!=settings.AASR_MODAL_REVISION:
            raise RuntimeError('Unexpected Long backend response')
        self.metadata={key:result.get(key) for key in ('model','gpu','precision','model_load_seconds','inference_ms','cache_only')}
        return ProviderResponse(result['text'],result['input_tokens'],result['output_tokens'],MODEL)


class AdaptiveRuntime:
    def __init__(self):
        self.short=NeuralShortParser(settings.AASR_SHORT_DIR)
        self.router=RouterV2(settings.AASR_ROUTER_PATH)
        self.lock=threading.Lock()
        self.short.torch.set_num_threads(2)
        # Read threshold from the frozen artifact, never a demo override.
        self.threshold=float(self.router.threshold)

    def short_and_route(self,question,context):
        from threadpoolctl import threadpool_limits
        captured=[]
        with self.lock,threadpool_limits(limits=1):
            hook=self.short.model.register_forward_hook(lambda _m,_i,out:captured.append(out.detach().softmax(-1)[0].cpu().tolist()))
            try:prediction=self.short.predict(question,context)
            finally:hook.remove()
            features=safe_features(question,context,prediction,captured[0])
            features['operation_probabilities']=dict(zip(OPERATIONS,captured[0]))
            probability=float(self.router.long_probabilities([features])[0])
        return prediction,('LONG' if probability>=self.threshold else 'SHORT'),probability

    def long(self,question,context):
        provider=ModalProvider()
        result=LongParser(provider).parse(question,context)
        result.backend_metadata=getattr(provider,'metadata',{})
        return result


@lru_cache(maxsize=1)
def get_runtime():return AdaptiveRuntime()
