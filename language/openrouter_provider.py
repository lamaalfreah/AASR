"""Exact Qwen3-4B free provider; local credentials and durable per-call recovery."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError,URLError
from urllib.request import Request,build_opener,HTTPRedirectHandler
from .long_parser import ProviderResponse

MODEL='qwen/qwen3-4b:free'
BASE_URL='https://openrouter.ai/api/v1'


class ProviderStopped(RuntimeError):
    def __init__(self,code,message):
        super().__init__(message);self.code=code


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ProviderStopped('redirect','Provider redirect refused')


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def load_local_key(path):
    """Load only this variable; no shell sourcing, interpolation, printing, or logging."""
    path=Path(path)
    if path.is_file():
        values=[]
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            name,sep,value=line.strip().partition('=')
            if sep and name.removeprefix('export ').strip()=='OPENROUTER_API_KEY':
                value=value.strip()
                if value[:1] in ('\"',"'"):
                    if len(value)<2 or value[-1]!=value[0]:raise ProviderStopped('configuration','Invalid local key quoting')
                    value=value[1:-1]
                else:value=value.split(' #',1)[0].strip()
                values.append(value)
        if len(values)>1:raise ProviderStopped('configuration','Duplicate local key definitions')
        if values:os.environ['OPENROUTER_API_KEY']=values[0]
    if not os.environ.get('OPENROUTER_API_KEY','').strip():
        raise ProviderStopped('configuration','OPENROUTER_API_KEY not visible')


class OpenRouterProvider:
    def __init__(self,checkpoint_dir,env_file):
        load_local_key(env_file)
        self._key=os.environ['OPENROUTER_API_KEY']
        self.checkpoint_dir=Path(checkpoint_dir)
        self.verified=False;self.case_id=None;self.actual_calls=0;self.replayed_calls=0
        self.last_attempt=0.0

    def request(self,path,body=None):
        req=Request(BASE_URL+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),
                    headers={'Authorization':'Bearer '+self._key,'Content-Type':'application/json'},
                    method='GET' if body is None else 'POST')
        try:
            with build_opener(NoRedirect).open(req,timeout=45) as response:return json.load(response)
        except HTTPError as exc:
            # Never expose the body; it can echo request/account information.
            code='rate_limit' if exc.code==429 else 'model_unavailable' if exc.code in (404,503) else 'authentication' if exc.code in (401,403) else 'provider_http'
            raise ProviderStopped(code,f'OpenRouter HTTP {exc.code}; stopped with progress preserved') from None
        except (URLError,TimeoutError,OSError):
            raise ProviderStopped('connection','OpenRouter connection failed; no automatic inference retry') from None
        except (ValueError,TypeError):
            raise ProviderStopped('invalid_envelope','Provider returned invalid JSON') from None

    def verify_model(self):
        self.request('/key')  # Authenticated credential check; account fields are not retained.
        models=self.request('/models').get('data',[])
        entry=next((m for m in models if m.get('id')==MODEL),None)
        if entry is None:
            raise ProviderStopped('model_unavailable','Exact qwen/qwen3-4b:free is absent from the current OpenRouter model catalogue; no substitution')
        pricing=entry.get('pricing',{})
        if any(float(pricing.get(k,0))!=0 for k in ('prompt','completion','request')):
            raise ProviderStopped('pricing','Requested free model reports nonzero inference pricing; stopped')
        self.verified=True
        return {'provider':'OpenRouter','model':MODEL,'base_url':BASE_URL,'key_visible':True,
                'authentication_verified':True,'exact_model_listed':True,'pricing':pricing,
                'supported_parameters':entry.get('supported_parameters',[]),
                'note':'Catalogue listing is not a guarantee of live endpoint capacity. Smoke test must succeed before EVAL.'}

    def begin_case(self,case_id):self.case_id=case_id

    def complete(self,messages):
        if not self.verified:raise ProviderStopped('not_verified','Verify exact model before inference')
        if not self.case_id:raise ProviderStopped('missing_case','Set checkpoint case before inference')
        body={'model':MODEL,'messages':messages,'temperature':0,'max_tokens':900,
              'reasoning':{'enabled':False},'response_format':{'type':'json_object'},
              'provider':{'require_parameters':True,'allow_fallbacks':False}}
        digest=hashlib.sha256(json.dumps({'case_id':self.case_id,'body':body},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        path=self.checkpoint_dir/(digest+'.json')
        if path.exists():
            saved=json.loads(path.read_text());self.replayed_calls+=1
        else:
            # Space calls at <= ~15/minute; any 429 stops, including daily limits.
            delay=max(0,4-(time.monotonic()-self.last_attempt))
            if delay:time.sleep(delay)
            self.last_attempt=time.monotonic();began=time.perf_counter()
            self.actual_calls+=1
            data=self.request('/chat/completions',body)
            saved={'case_id':self.case_id,'request_sha256':digest,'requested_model':MODEL,
                   'latency_ms':(time.perf_counter()-began)*1000,
                   'provider_model':data.get('model'),'usage':data.get('usage',{}),
                   'choices':data.get('choices',[]),'response_id':data.get('id')}
            # Persist every successful HTTP/JSON response BEFORE schema validation,
            # including truncated/invalid model output, so retries survive restarts.
            atomic_json(path,saved)
        # OpenRouter may omit :free in the canonical response ID; same exact 4B
        # model only. Requests always pin :free and preflight requires zero price.
        if saved['provider_model'] not in (MODEL,'qwen/qwen3-4b'):
            raise ProviderStopped('model_mismatch','Response model is not the requested Qwen3-4B; stopped')
        try:text=saved['choices'][0]['message']['content']
        except (KeyError,IndexError,TypeError):raise ProviderStopped('invalid_envelope','Missing response content; checkpoint retained') from None
        usage=saved['usage']
        return ProviderResponse(text if isinstance(text,str) else '',usage.get('prompt_tokens'),usage.get('completion_tokens'),saved['provider_model'])
