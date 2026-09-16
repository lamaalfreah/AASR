"""Optional browser checks against a running local server; only synthetic demo data.

python -m dashboard.browser_smoke [--long-response /path/to/saved-production-response.json]
Requires Playwright + Chromium. Does not request Long inference: optional Long UI
coverage replays a saved production response; error responses are test fixtures.
"""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--long-response',type=Path)
    args=parser.parse_args()
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':1050})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url,wait_until='networkidle')
        expect(page.locator('#location-catalog')).to_have_count(1)
        for question,answer in [
            ('ما أقرب مستشفى إلى «مركز الحي»؟','مستشفى النخيل'),
            ('كم عدد الصيدليات ضمن 2 كم من «مركز الحي»؟','2'),
            ('ما اتجاه «مدرسة الرواد» بالنسبة إلى «مركز الحي»؟','شمال'),
        ]:
            page.locator('#question-input').fill(question)
            with page.expect_response('**/api/analyze/') as response:
                page.locator('#analyze-button').click()
            data=response.value.json()
            assert data['status']=='success' and data['route']=='SHORT',data
            expect(page.locator('#answer-title')).to_have_text(answer)
            expect(page.locator('#route-status')).to_contain_text('SHORT')
            expect(page.locator('#trace-list li')).to_have_count(4)
            expect(page.locator('#analyze-button')).to_be_enabled()
            print('PASS real Django/Short/Router/Geo/UI:',question,flush=True)
        if page.evaluate('Boolean(window.L)'):
            assert page.locator('#map .leaflet-interactive').count()>=2
            print('PASS selected map markers',flush=True)
        else:
            raise AssertionError('Leaflet unavailable: cannot verify map markers')
        if args.long_response:
            saved=json.loads(args.long_response.read_text())
            assert saved['route']=='LONG'
            page.route('**/api/analyze/',lambda route:route.fulfill(json=saved))
            page.locator('#analyze-button').click()
            expect(page.locator('#route-status')).to_contain_text('LONG')
            expect(page.locator('#answer-title')).to_have_text(saved['answer']['text'])
            expect(page.locator('#trace-list li')).to_have_count(5)
            page.unroute('**/api/analyze/')
            print('PASS saved real Long response UI replay (no generation)',flush=True)
        for status,code in [('ambiguous',200),('needs_clarification',200),('unavailable',503),('busy',429)]:
            page.route('**/api/analyze/',lambda route,_request,s=status,c=code:route.fulfill(status=c,json={'status':s,'message':s,'answer':None}))
            page.locator('#analyze-button').click()
            expect(page.locator('#answer-title')).to_have_text(status)
            expect(page.locator('#metric-3-value')).to_have_text('—')
            expect(page.locator('#analyze-button')).to_be_enabled()
            assert page.locator('#map .leaflet-interactive').count()==0
            page.unroute('**/api/analyze/')
            print('PASS UI fixture:',status,flush=True)
        page.route('**/api/analyze/',lambda route:route.abort())
        page.locator('#analyze-button').click()
        expect(page.locator('#answer-symbol')).to_have_text('!')
        expect(page.locator('#analyze-button')).to_be_enabled()
        page.unroute('**/api/analyze/')
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
        assert not errors,errors
        print('PASS network failure, mobile layout, no JavaScript exceptions',flush=True)
        fallback=browser.new_page()
        fallback.route('**/*leaflet*',lambda route:route.abort())
        fallback.goto(args.url,wait_until='networkidle')
        expect(fallback.locator('#map')).to_contain_text('الخريطة غير متاحة')
        print('PASS map library unavailable state',flush=True)
        browser.close()


if __name__=='__main__':
    main()
