#!/usr/bin/env python3
"""Credential-safe exact OpenRouter model discovery. No generation calls."""
import json
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from language.openrouter_provider import OpenRouterProvider,ProviderStopped,atomic_json


def main():
    out=BASE/'experiments/E5_lightweight_adaptive/step2_long'
    try:
        provider=OpenRouterProvider(out/'api_calls',BASE/'.env')
        result=provider.verify_model();result['status']='verified'
    except ProviderStopped as e:
        result={'provider':'OpenRouter','model':'qwen/qwen3-4b:free','status':'stopped',
                'reason_code':e.code,'reason':str(e),'inference_calls':0}
    atomic_json(out/'provider_preflight.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
