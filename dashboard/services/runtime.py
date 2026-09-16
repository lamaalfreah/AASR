"""Lazy frozen CPU models and a Long-only Modal provider. No evaluation imports."""
import hashlib
import json
import threading
import uuid
from functools import lru_cache
from django.conf import settings
from spatial.schema import OPERATIONS
from .features import safe_features


class Unavailable(RuntimeError):
    pass


def verify_frozen():
    manifest = json.loads((settings.BASE_DIR/'dashboard/data/production_freeze.json').read_text())
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((settings.BASE_DIR/name).read_bytes()).hexdigest() != expected:
            raise Unavailable('Frozen artifact hash mismatch: '+name)


class Runtime:
    def __init__(self):
        verify_frozen()
        from language.neural_short import NeuralShortParser
        from language.router_v2 import RouterV2
        self.short = NeuralShortParser(settings.BASE_DIR/'experiments/E5_lightweight_adaptive/step2_neural_short')
        self.router = RouterV2(settings.BASE_DIR/'experiments/E5_lightweight_adaptive/router_v2/router_v2.joblib')
        if self.router.threshold != .91 or len(self.router.ensemble.components) != 3:
            raise Unavailable('Frozen Router configuration mismatch')
        self.lock = threading.Lock()
        self.short.torch.set_num_threads(settings.AASR_CPU_THREADS)

    def predict_short(self, question, context):
        # Capture exactly the logits from the frozen predict() call; no second inference.
        with self.lock:
            captured=[]
            def capture(_module, _inputs, output):
                captured.append(output.detach().softmax(-1)[0].cpu().tolist())
            handle=self.short.model.register_forward_hook(capture)
            try:
                prediction=self.short.predict(question, context)
            finally:
                handle.remove()
        probabilities=captured[0]
        features=safe_features(question, context, prediction, probabilities)
        features['operation_probabilities']=dict(zip(OPERATIONS, probabilities))
        return prediction, features

    def route(self, features):
        from threadpoolctl import threadpool_limits
        with self.lock, threadpool_limits(limits=1):
            probability=float(self.router.long_probabilities([features])[0])
        return ('LONG' if probability >= .91 else 'SHORT'), probability

    def predict_long(self, question, context):
        from language.long_parser import LongParser
        return LongParser(ModalProvider()).parse(question, context)


class ModalProvider:
    def __init__(self):
        if not settings.AASR_MODAL_APP:
            raise Unavailable('Long backend is not configured')
        self.case=uuid.uuid4().hex

    def complete(self, messages):
        import modal
        from language.long_parser import ProviderResponse
        from language.modal_checkpoint import digest, GENERATION, MODEL
        revision='1cfa9a7208912126459214e8b04321603b3df60c'
        model=modal.Cls.from_name(settings.AASR_MODAL_APP, 'QwenLong')(revision=revision)
        key=digest(dict(namespace='production',case=self.case,messages=messages,revision=revision,generation=GENERATION))
        call=model.generate.spawn(messages,key)
        try:
            saved=call.get(timeout=settings.AASR_LONG_TIMEOUT)
        except TimeoutError:
            call.cancel()
            raise Unavailable('Long request timed out') from None
        if saved.get('request_digest')!=key or saved.get('model')!=MODEL or saved.get('revision')!=revision or not saved.get('cache_only') or saved.get('gpu_count')!=1 or saved.get('gpu')!='NVIDIA L4' or saved.get('precision')!='torch.bfloat16':
            raise Unavailable('Long backend identity mismatch')
        return ProviderResponse(saved['text'],saved['input_tokens'],saved['output_tokens'],MODEL)


@lru_cache(maxsize=1)
def get_runtime():
    return Runtime()
