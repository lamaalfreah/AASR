"""Environment-only DashScope client with exact-model availability gate."""
from __future__ import annotations
import json
import os
import re
from urllib.error import HTTPError,URLError
from urllib.parse import urlsplit,urlunsplit,urlencode
from urllib.request import Request,build_opener,HTTPRedirectHandler
from .long_parser import ProviderResponse

MODEL='qwen3-4b'  # Explicit user choice. Never substitute or alias a different model.


class ProviderUnavailable(RuntimeError):pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise ProviderUnavailable('Provider redirect refused; verify configured endpoint')


class QwenProvider:
    def __init__(self):
        self._key=os.environ.get('DASHSCOPE_API_KEY')
        self.base_url=os.environ.get('DASHSCOPE_BASE_URL','').rstrip('/')
        self.models_url=os.environ.get('DASHSCOPE_MODELS_URL','')
        self.verified=False
        self.availability_calls=0
        if not self._key or not self.base_url:
            raise ProviderUnavailable('Missing DASHSCOPE_API_KEY and/or DASHSCOPE_BASE_URL; account/region access unverified')
        parts=urlsplit(self.base_url)
        if parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ProviderUnavailable('Expected a credential-free HTTPS base URL')
        if not self.models_url:
            if not parts.path.endswith('/compatible-mode/v1'):
                raise ProviderUnavailable('Set DASHSCOPE_MODELS_URL for a nonstandard regional endpoint')
            self.models_url=urlunsplit((parts.scheme,parts.netloc,'/api/v1/models','',''))
        models=urlsplit(self.models_url)
        if models.scheme!='https' or models.netloc!=parts.netloc or models.query or models.fragment:
            raise ProviderUnavailable('Model discovery must use the same HTTPS provider host')

    def _request(self,url,body=None):
        request=Request(url,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),
            headers={'Authorization':'Bearer '+self._key,'Content-Type':'application/json'},
            method='GET' if body is None else 'POST')
        try:
            with build_opener(NoRedirect).open(request,timeout=45) as response:
                return json.load(response)
        except HTTPError as exc:
            # Do not echo provider bodies, request headers, or credential-bearing objects.
            raise ProviderUnavailable('Provider HTTP '+str(exc.code)+'; no model substitution') from None
        except (URLError,TimeoutError,OSError):
            raise ProviderUnavailable('Provider connection failed; no model substitution') from None

    def verify_model(self):
        self.availability_calls+=1
        data=self._request(self.models_url+'?'+urlencode({'model':MODEL,'page_size':100,'page_no':1}))
        models=data.get('output',{}).get('models',data.get('data',[]))
        ids=sorted({m.get('model',m.get('id','')) for m in models if isinstance(m,dict)})
        if MODEL not in ids:
            self.availability_calls+=1
            alternatives=self._request(self.models_url+'?'+urlencode({'name':'qwen3','page_size':100,'page_no':1}))
            models=alternatives.get('output',{}).get('models',alternatives.get('data',[]))
            compatible=sorted({m.get('model',m.get('id','')) for m in models if isinstance(m,dict)
                and re.fullmatch(r'[A-Za-z0-9_./-]{1,100}',m.get('model',m.get('id','')))
                and 'qwen3' in m.get('model',m.get('id','')).lower()
                and '4b' in m.get('model',m.get('id','')).lower()})
            listed=', '.join(compatible) if compatible else 'none returned on the first 100 catalogue matches'
            raise ProviderUnavailable('Exact qwen3-4b not listed for configured account/region. '
                'Qwen3 4B catalogue options: '+listed+'. No challenge calls made; explicit model/deployment choice required.')
        self.verified=True
        return {'requested_model':MODEL,'exact_model_listed':True,'account_region_verified':True,
                'verification_method':'Authenticated regional model-list request; inference authorization still subject to provider response',
                'base_url':self.base_url,'availability_calls':self.availability_calls}

    def complete(self,messages):
        if not self.verified:
            raise ProviderUnavailable('Call verify_model before any inference request')
        data=self._request(self.base_url+'/chat/completions',{
            'model':MODEL,'messages':messages,'temperature':0,'max_tokens':900,
            'enable_thinking':False,'response_format':{'type':'json_object'}})
        model=data.get('model')
        if model and model!=MODEL:
            raise ProviderUnavailable('Provider returned a different model ID; stopped without substitution')
        try:text=data['choices'][0]['message']['content']
        except (KeyError,IndexError,TypeError):raise ProviderUnavailable('Malformed provider envelope') from None
        if not isinstance(text,str):raise ProviderUnavailable('Missing provider text content')
        usage=data.get('usage',{})
        return ProviderResponse(text,usage.get('prompt_tokens'),usage.get('completion_tokens'),model)
