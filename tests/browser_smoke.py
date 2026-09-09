"""Browser tests using an explicitly labelled synthetic body fixture.
No screenshot from this script is evidence of a MuJoCo run.
"""
from pathlib import Path
import os
import subprocess,sys,time,json,urllib.request
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
results=[]
validation_mode="normal localhost fixture"

def check(name,condition):
    results.append(dict(name=name,passed=bool(condition)))
    if not condition:raise AssertionError(name)

if __name__=='__main__':
    log=open(ROOT/'tests/fixture_server.log','w')
    proc=subprocess.Popen([sys.executable,str(ROOT/'tests/fixture_server.py')],cwd=ROOT,stdout=log,stderr=log)
    errors=[]
    try:
        for _ in range(50):
            try:urllib.request.urlopen('http://127.0.0.1:8766/api/doctor',timeout=.2);break
            except Exception:time.sleep(.1)
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
            page=browser.new_page(viewport={'width':1500,'height':1040},device_scale_factor=1,accept_downloads=True)
            page.on('pageerror',lambda e:errors.append(str(e)))
            try:
                page.goto('http://127.0.0.1:8766/?software=1',wait_until='domcontentloaded')
            except Exception as nav_error:
                if 'ERR_BLOCKED_BY_ADMINISTRATOR' not in str(nav_error):raise
                validation_mode='in-process UI fixture; managed browser blocked localhost navigation'
                sys.path.insert(0,str(ROOT))
                from flylab.engine import Dispatcher
                from tests.fixture_body import FixtureBody
                from flylab import PROTOCOL
                dispatcher=Dispatcher(FixtureBody)
                counter=[0]
                def bridge(op,payload):
                    counter[0]+=1
                    return dispatcher.handle(dict(protocol=PROTOCOL,requestId=counter[0],op=op,payload=payload))['result']
                page.expose_function('_fixtureRpc',bridge)
                html=(ROOT/'FLY_LAB_B.html').read_text()
                injection="""globalThis.FLY_FORCE_SOFTWARE=true;
                Fly.BTransport=class{constructor(){this.name='UI TEST DOUBLE';}async connect(){document.body.classList.add('test-fixture');document.getElementById('verification-banner').textContent='UI TEST FIXTURE · 합성 테스트 상태 · 실제 물리 실행 화면 아님';}request(op,p={}){return window._fixtureRpc(op,p);}close(){}};
                """
                marker='(function (F) {\n    \'use strict\';\n    const $ = id'
                if marker not in html:raise RuntimeError('UI test injection point not found')
                html=html.replace(marker,injection+marker,1)
                page.set_content(html,wait_until='domcontentloaded')
            page.wait_for_function('window.flyLab?.ready === true',timeout=15000)
            page.evaluate('flyLab.setPaused(true)')
            page.evaluate('flyLab.advanceSteps(220)')
            check('fixture visibly labelled','UI TEST FIXTURE' in page.locator('#verification-banner').inner_text())
            check('170 neural nodes',page.locator('#node-count').inner_text()=='170')
            check('42 telemetry rows',page.locator('#joint-rows tr').count()==42)
            check('6 contacts bars',page.locator('#feet-bars>div').count()==6)
            check('backend gate dismissed',not page.locator('#backend-gate').is_visible())
            check('B header',page.locator('.stage-badge').inner_text()=='B')
            check('walk only',page.locator('#mode-select option').count()==1)
            check('clock matches',page.evaluate('Math.abs(flyLab.frame.simTime-flyLab.frame.physics.physicsTime)<1e-6'))
            page.locator('#task-select').select_option('heading');page.wait_for_function('flyLab.frame.config.task==="heading"')
            check('task selection',page.evaluate('flyLab.frame.config.task==="heading"'))
            page.locator('#search').fill('model:DN');check('neuron search','2 / 170' in page.locator('#visible-count').inner_text())
            page.locator('#search').fill('');page.evaluate('flyLab.choose("model:DN:0")')
            page.locator('#suppress-btn').click();page.wait_for_function('flyLab.frame.neural.lesions.includes("model:DN:0")')
            check('suppress action',True);page.locator('#release-btn').click();page.wait_for_function('flyLab.frame.neural.lesions.length===0')
            check('release action',True)
            page.locator('#stim-btn').click();page.wait_for_function('flyLab.frame.neural.stimulated.includes("model:DN:0")')
            check('stimulate action',True)
            cp=page.evaluate('flyLab.checkpoint()');page.evaluate('flyLab.advanceSteps(30)');page.evaluate('(cp)=>flyLab.restore(cp)',cp)
            check('checkpoint restore',page.evaluate('flyLab.frame.tick')==cp['tick'])
            page.locator('#motor-toggle').uncheck();page.wait_for_function('flyLab.frame.config.motorCoupled===false');check('motor switch',True)
            page.locator('#motor-toggle').check();page.wait_for_function('flyLab.frame.config.motorCoupled===true')
            page.locator('#push-btn').click();check('force command delivered','push' in page.locator('#events').inner_text())
            for id,suffix in [('csv-btn','.csv'),('physics-csv-btn','.csv'),('checkpoint-btn','.json'),('graph-download','.json'),('replay-export-btn','.json'),('photo-btn','.png')]:
                with page.expect_download() as info:page.locator('#'+id).click()
                download=info.value;check('download '+id,download.suggested_filename.endswith(suffix))
                raw=Path(download.path()).read_bytes()
                if suffix=='.json': assert isinstance(json.loads(raw),dict)
                elif suffix=='.csv':
                    import csv,io
                    cols=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
                    assert len(cols[0])==(135 if id=='physics-csv-btn' else 177)
                else: assert raw[:8]==b'\x89PNG\r\n\x1a\n'
            page.locator('#experiment-kind').select_option('motor-off');page.locator('#paired-btn').click();page.wait_for_selector('#paired-dialog[open]',timeout=30000)
            check('paired dialog',page.locator('#paired-metrics').inner_text().find('대조군')>=0)
            page.locator('[data-close="paired-dialog"]').click()
            page.locator('#about-btn').click();check('model card',page.locator('#about-dialog').is_visible());page.locator('[data-close="about-dialog"]').click()
            page.locator('#architecture-btn').click();check('architecture dialog',page.locator('#architecture-dialog').is_visible());page.locator('[data-close="architecture-dialog"]').click()
            page.evaluate('flyLab.advanceSteps(150)');page.evaluate('window.scrollTo(0,0)')
            page.screenshot(path=str(ROOT/'tests/UI_FIXTURE_desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
            check('mobile no horizontal overflow',page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+2'))
            page.screenshot(path=str(ROOT/'tests/UI_FIXTURE_mobile.png'),full_page=True)
            check('no JS errors',not errors)
            browser.close()
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill()
        log.close();(ROOT/'tests/browser_result.json').write_text(json.dumps(dict(physicalValidation=False,fixture=True,mode=validation_mode,tests=results,errors=errors),ensure_ascii=False,indent=2))
    print(json.dumps(dict(passed=len(results),errors=errors,physicalValidation=False),ensure_ascii=False))
