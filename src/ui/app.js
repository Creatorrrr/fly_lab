(function (F) {
    'use strict';
    const $ = id => document.getElementById(id), DEG = 180 / Math.PI;
    const baseGraph = JSON.parse($('fly-data').textContent);
    let transport = null, frame = null, catalog = [], graph = baseGraph, monitor = null, view = null;
    let paused = false, busy = false, locked = false, speed = 1, accumulator = 0, lastWall = 0, lastHUD = 0, lastSample = -1, history = [], selected = '', pins = [], placement = null, paired = null, rafId = 0;
    let selectedFilter = [], lastValues = {}, toastTimer = 0, fps = 0, fpsFrames = 0, fpsStart = performance.now();
    const text = (id, s) => { $(id).textContent = s; }, button = (id, active) => $(id).classList.toggle('active', !!active);
    function toast(s, error = false) { const e = $('toast'); e.textContent = s; e.classList.toggle('error', error); e.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => e.classList.remove('show'), 6000); }
    function download(name, content, type = 'application/json') { const blob = content instanceof Blob ? content : new Blob([content], { type }), url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
    const stamp = () => `seed${frame?.seed ?? $('seed').value}_tick${frame?.tick ?? 0}`;
    function jsonDownload(name, data) { download(name, JSON.stringify(data, null, 2), 'application/json;charset=utf-8'); }
    function setPaused(value) { paused = !!value; accumulator = 0; $('play-btn').textContent = paused ? '▶ 재생' : 'Ⅱ 일시정지'; $('play-btn').setAttribute('aria-pressed', String(paused)); button('play-btn', paused); }
    function accept(f) { frame = f; if(f.physics?.fault){setPaused(true);toast(f.physics.fault,true);} updatePhysics(f); view.setFrame(f); lastValues = Object.fromEntries(f.neural.ids.map((id, i) => [id, f.neural.values[i]])); const s = Math.floor(f.simTime * 10 + 1e-5); if (s !== lastSample) {
        history.push({ t: f.simTime, values: { ...lastValues } });
        lastSample = s;
        while (history.length > 1201 || history.length && history[0].t < f.simTime - 120)
            history.shift();
    } updateHUD(); return f; }
    function configureReady(r) {
        catalog = r.catalog;
        graph = r.graph;
        monitor = new F.Monitor.NeuralMonitor(catalog, graph);
        history = [];
        lastSample = -1;
        view.trail.clear();
        view.lastTick = -1;
        const all = new Set(catalog.map(n => n.id));
        selected = all.has(selected) ? selected : (catalog.find(n => n.type === 'EPG') || catalog[0]).id;
        pins = pins.filter(id => all.has(id));
        if (!pins.length)
            pins = [selected, 'model:HD:0', 'model:PFL3-L:0', 'model:PFL3-R:0', 'model:DN:0', 'model:DN:1'].filter(id => all.has(id));
        selectedFilter = catalog;
        $('seed').value = String(r.frame.seed);
        const real = catalog.filter(n => n.origin === 'anatomical').length, model = catalog.length - real;
        text('node-count', String(catalog.length));
        text('real-count', String(real));
        text('model-count', String(model));
        text('scope-text', `${graph.manifest?.userImport ? '사용자 자료 · 출처 미검증' : 'hemibrain 발췌'}  ${real} IDs · ${graph.edges.length} 연결  +  ${model} 설계 노드`);
        text('selected-origin', '');
        filterCatalog();
        accept(r.frame);
        syncControls();
        renderLegend();
        text('runtime', `${transport.name} · ${view.engine || '3D'} · 신경 5 ms · 물리 0.1 ms`);
        return r;
    }
    function syncControls() { if (!frame)
        return; const c = frame.config; $('mode-select').value = c.mode; $('task-select').value = c.task; $('goal-slider').value = String(Math.round(F.wrap(c.goalAngle) * DEG)); $('altitude-slider').value = String(c.altitude); text('goal-output', Math.round(c.goalAngle * DEG) + '°'); text('altitude-output', c.altitude.toFixed(1) + ' mm'); $('cue-toggle').checked = frame.world.cueOn; $('food-toggle').checked = frame.world.foodOn; $('motor-toggle').checked=c.motorCoupled; $('friction-slider').value=String(c.friction); text('friction-output',c.friction+'×'); }
    async function command(type, payload = {}) { const f = await transport.request('command', { type, payload }); accept(f); syncControls(); return f; }
    async function exclusive(fn) { if (locked)
        throw Error('다른 파일/실험 작업이 진행 중입니다.'); locked = true; try {
        while (busy)
            await new Promise(resolve => setTimeout(resolve, 5));
        return await fn();
    }
    finally {
        locked = false;
        accumulator = 0;
        lastWall = performance.now();
    } }
    async function reset() { return exclusive(async () => { const seed = Number($('seed').value); if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
        throw Error('seed는 0~4294967295 정수여야 합니다.'); const r = await transport.request('init', { graph, seed, config: frame?.config || {} }); configureReady(r); view.track(); toast('동일한 seed와 기본 환경에서 다시 시작했습니다.'); return r; }); }
    function choose(id) { if (!catalog.some(n => n.id === id))
        return; selected = id; updateSelection(); drawMonitors(); }
    function filterCatalog() { const q = $('search').value.trim().toLowerCase(), group = $('group-select').value; selectedFilter = catalog.filter(n => (group === 'all' || group === 'anatomical' && n.origin === 'anatomical' || n.type === group) && (!q || (n.id + ' ' + n.displayName + ' ' + n.name).toLowerCase().includes(q))); text('visible-count', `${selectedFilter.length} / ${catalog.length}`); drawMonitors(); }
    function updateSelection() { if (!frame)
        return; const n = catalog.find(n => n.id === selected); if (!n)
        return; const real = n.origin === 'anatomical'; text('selected-origin', real ? 'ACTUAL ID · 계산된 활성도' : 'ENGINEERED · 설계 계산 노드'); $('selected-origin').classList.toggle('modeled', !real); text('selected-name', n.displayName || n.name); text('selected-id', n.id); text('selected-value', (lastValues[selected] ?? 0).toFixed(3)); const ins = graph.edges.filter(e => e.post === selected).length, outs = graph.edges.filter(e => e.pre === selected).length; text('selected-info', real ? `${n.name} · 발췌 입력 ${ins} / 출력 ${outs}` : `${n.type} · 해부학적 ID 아님 · 모델 내부 연산`); const lesion = frame.neural.lesions.includes(selected); $('suppress-btn').textContent = lesion ? '억제 해제' : '활성 억제'; button('suppress-btn', lesion); $('pin-btn').textContent = pins.includes(selected) ? '고정 해제' : '그래프 고정'; button('pin-btn', pins.includes(selected)); button('left-lesion-btn', frame.neural.lesions.some(id => id.startsWith('model:PFL3-L:'))); }
    function updateHUD() { if (!frame)
        return; const f = frame; const t = f.simTime; text('time', `${Math.floor(t / 60).toString().padStart(2, '0')}:${(t % 60).toFixed(1).padStart(4, '0')}`); text('behavior', f.neural.recoveryActive ? 'BACK OFF · 접촉 후진' : f.neural.avoidanceActive ? 'AVOID · 회피 회전' : f.neural.behavior === 'dwell' ? 'DWELL · 머무름' : f.config.task === 'heading' ? 'HEADING · 방향 유지' : 'EXPLORE · 탐색'); text('distance', f.body.travel.toFixed(1)); text('height', f.body.position[1].toFixed(2)); text('velocity', f.body.speed.toFixed(2)); text('yaw-output', (f.neural.output.yawRate * DEG).toFixed(1)); text('contacts', String(f.body.collisions)); text('confidence', (f.neural.confidence * 100).toFixed(0) + '%'); text('estimated-heading', Math.round(F.wrap(f.neural.heading) * DEG) + '°'); text('goal-heading', Math.round(F.wrap(f.neural.goal) * DEG) + '°'); text('true-heading', Math.round(F.wrap(f.body.yaw) * DEG) + '°'); text('cue-state', f.world.cueOn ? 'ON · 시각 기준' : 'OFF · 암전'); text('odor-state', f.world.foodOn ? (f.sensors?.odor?.reduce((a, b) => a + b, 0) / 2 || 0).toFixed(2) : 'OFF'); text('recording-info', `계산기 기록 10 Hz · 최근 ${Math.min(120, t).toFixed(0)} s · ${f.recording.samples} samples`); text('events', f.events.slice(-4).map(e => `[${e.time.toFixed(1)}s] ${e.message}`).reverse().join('   /   ') || '실험이 시작되었습니다. 뉴런을 선택하거나 환경을 변경해 보세요.'); updateSelection(); }
    function updatePhysics(f){
      if(!f.physics)return;const p=f.physics;
      text('physics-status',p.testDouble?'TEST FIXTURE':p.backend);
      text('physical-clock',p.physicsTime.toFixed(3)+' s');
      text('descending-value',p.descending.map(x=>x.toFixed(3)).join(' / '));
      text('native-count','42 active DOFs · 6 feet');
      text('physics-fault',p.fault || (f.body.contact ? '벽·장애물 또는 몸통 접촉 감지' : f.neural.avoidanceActive ? '회피 회전 중 · 전방 여유 공간 확인' : '수치 이상 감지 없음 · 행동 성공 판정은 아님'));
      const feet=$('feet-bars');if(!feet.children.length){['LF','LM','LH','RF','RM','RH'].forEach(n=>{const cell=document.createElement('div');cell.innerHTML='<b>'+n+'</b><meter min="0" max="3" value="0"></meter><span>0.00 BW</span>';feet.append(cell);});}
      Array.from(feet.children).forEach((c,i)=>{const v=p.contactsBW[i];c.querySelector('meter').value=Math.min(3,v);c.querySelector('span').textContent=v.toFixed(2)+' BW';c.classList.toggle('touching',v>.01);});
      const rows=$('joint-rows');if(rows.children.length!==p.jointNames.length){rows.replaceChildren();p.jointNames.forEach((name)=>{const r=document.createElement('tr');for(let k=0;k<4;k++)r.append(document.createElement('td'));r.children[0].textContent=name;rows.append(r);});}
      Array.from(rows.children).forEach((r,i)=>{r.children[1].textContent=p.jointAngles[i].toFixed(3);r.children[2].textContent=p.jointTargets[i].toFixed(3);r.children[3].textContent=p.actuatorForces[i].toFixed(3);});
    }
    function drawMonitors() { if (!frame || !monitor)
        return; F.Monitor.compass($('compass'), frame); F.Monitor.sensors($('sensors'), frame); monitor.network($('network'), frame, selected); monitor.heatmap($('heatmap'), frame, selected, selectedFilter); F.Monitor.timeline($('timeline'), history, pins, catalog, frame); if ($('paired-dialog').open && paired)
        F.Monitor.pairedChart($('paired-chart'), paired); }
    function renderLegend() { const parent = $('trace-legend'); parent.replaceChildren(); pins.forEach((id, i) => { const n = catalog.find(n => n.id === id), b = document.createElement('button'); b.type = 'button'; b.className = 'trace-chip'; b.style.setProperty('--trace-color', F.Monitor.palette[i]); b.textContent = n?.displayName || id; b.title = '클릭: 이 노드 선택'; b.onclick = () => choose(id); parent.append(b); }); }
    function on(id, event, fn) { $(id).addEventListener(event, async (e) => { try {
        await fn(e);
    }
    catch (err) {
        toast(err.message || String(err), true);
    } }); }
    function openDialog(id) { const d = $(id); if (!d.open)
        d.showModal(); requestAnimationFrame(drawMonitors); }
    function initControls() {
        on('push-btn','click',()=>command('push',{bw:.5,duration:.05}));
        on('motor-toggle','change',()=>command('configure',{motorCoupled:$('motor-toggle').checked}));
        on('friction-slider','input',()=>text('friction-output',$('friction-slider').value+'×'));
        on('friction-slider','change',()=>command('configure',{friction:Number($('friction-slider').value)}));
        on('physics-csv-btn','click',()=>exclusive(async()=>download(`FLY_LAB_B_physics_${stamp()}.csv`,await transport.request('physicsCsv'),'text/csv;charset=utf-8')));
        on('native-preview-btn','click',()=>exclusive(async()=>{const r=await transport.request('preview');$('native-preview').src=r.image;$('native-preview').hidden=false;text('native-preview-caption',r.label+' · tick '+r.tick);}));

        on('play-btn', 'click', () => setPaused(!paused));
        on('step-btn', 'click', () => exclusive(async () => { setPaused(true); accept(await transport.request('advance', { steps: 10 })); drawMonitors(); }));
        on('speed-select', 'change', () => { speed = Number($('speed-select').value); accumulator = 0; });
        on('mode-select', 'change', () => command('configure', { mode: $('mode-select').value }));
        on('reset-btn', 'click', reset);
        on('seed-btn', 'click', reset);
        on('follow-btn', 'click', () => { view.track(); view.firstPerson = false; button('eye-btn', false); button('follow-btn', true); });
        on('home-btn', 'click', () => { view.home(); view.firstPerson = false; button('eye-btn', false); button('follow-btn', false); });
        on('top-btn', 'click', () => { view.top(); view.firstPerson = false; button('eye-btn', false); button('follow-btn', false); });
        on('eye-btn', 'click', () => { view.firstPerson = !view.firstPerson; button('eye-btn', view.firstPerson); text('camera-hint', view.firstPerson ? '개체 시점 · 실제 복안 영상 아님' : '드래그 회전 · 휠 확대 · F 추적'); });
        on('cue-toggle', 'change', () => command('cue', { enabled: $('cue-toggle').checked }));
        on('food-toggle', 'change', () => command('food', { enabled: $('food-toggle').checked }));
        on('task-select', 'change', () => command('configure', { task: $('task-select').value }));
        on('goal-slider', 'input', () => text('goal-output', $('goal-slider').value + '°'));
        on('goal-slider', 'change', () => command('configure', { goalAngle: Number($('goal-slider').value) / DEG }));
        on('altitude-slider', 'input', () => text('altitude-output', Number($('altitude-slider').value).toFixed(1) + ' mm'));
        on('altitude-slider', 'change', () => command('configure', { altitude: Number($('altitude-slider').value) }));
        on('search', 'input', filterCatalog);
        on('group-select', 'change', filterCatalog);
        for (const [id, method] of [['network', 'pickNetwork'], ['heatmap', 'pickHeat']])
            on(id, 'click', e => { const r = $(id).getBoundingClientRect(), hit = monitor[method](e.clientX - r.left, e.clientY - r.top); if (hit)
                choose(hit); });
        on('stim-btn', 'click', () => command('intervene', { kind: 'stimulate', ids: [selected], amplitude: .65, duration: 2 }));
        on('suppress-btn', 'click', () => command('intervene', { kind: frame.neural.lesions.includes(selected) ? 'release' : 'suppress', ids: [selected] }));
        on('left-lesion-btn', 'click', () => command('intervene', { kind: frame.neural.lesions.some(id => id.startsWith('model:PFL3-L:')) ? 'release' : 'suppress', ids: catalog.filter(n => n.type === 'PFL3-L').map(n => n.id) }));
        on('release-btn', 'click', () => command('releaseAll'));
        on('pin-btn', 'click', () => { if (pins.includes(selected))
            pins = pins.filter(id => id !== selected);
        else {
            if (pins.length >= 8)
                throw Error('그래프는 최대 8개입니다. 다른 고정을 해제해 주세요.');
            pins.push(selected);
        } renderLegend(); updateSelection(); drawMonitors(); });
        document.querySelectorAll('[data-place]').forEach(b => b.addEventListener('click', () => { placement = placement === b.dataset.place ? null : b.dataset.place; document.querySelectorAll('[data-place]').forEach(x => x.classList.toggle('active', x.dataset.place === placement)); $('placement-banner').hidden = !placement; text('placement-banner', placement ? `${b.textContent.trim()} · 3D 공간을 클릭해 배치 / Esc 취소` : ''); }));
        on('clear-added-btn', 'click', () => command('clearAdded'));
        const canvas = view.canvas;
        let drag = null;
        canvas.style.touchAction = 'none';
        canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, moved: false }; canvas.setPointerCapture(e.pointerId); });
        canvas.addEventListener('pointermove', e => { if (!drag)
            return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 5)
            drag.moved = true; if (!placement) {
            view.theta -= dx * .006;
            view.phi = F.clamp(view.phi + dy * .005, .035, 1.52);
            view.firstPerson = false;
            button('eye-btn', false);
        } drag.x = e.clientX; drag.y = e.clientY; });
        canvas.addEventListener('pointerup', async (e) => { const d = drag; drag = null; if (!d || d.moved || !placement)
            return; try {
            const r = canvas.getBoundingClientRect(), h = Number($('place-height').value);
            if (!Number.isFinite(h) || h < .2 || h > 14)
                throw Error('배치 높이는 0.2~14 mm여야 합니다.');
            view.camera();
            const p = view.pointOnPlane(e.clientX - r.left, e.clientY - r.top, h);
            if (!p)
                throw Error('이 시점에서는 배치 평면을 볼 수 없습니다. 전체 보기를 사용해 주세요.');
            await command('place', { kind: placement, position: p });
            toast('환경에 배치했습니다. Esc로 배치 모드를 종료합니다.');
        }
        catch (err) {
            toast(err.message, true);
        } });
        canvas.addEventListener('pointercancel', () => drag = null);
        canvas.addEventListener('wheel', e => { e.preventDefault(); view.distance = F.clamp(view.distance * Math.exp(e.deltaY * .001), 5, 125); }, { passive: false });
        on('csv-btn', 'click', () => exclusive(async () => download(`FLY_LAB_B_${stamp()}.csv`, await transport.request('csv'), 'text/csv;charset=utf-8')));
        on('checkpoint-btn', 'click', () => exclusive(async () => jsonDownload(`FLY_LAB_B_checkpoint_${stamp()}.json`, await transport.request('checkpoint'))));
        on('graph-download', 'click', () => exclusive(async () => jsonDownload('FLY_LAB_B_connectome.json', await transport.request('graph'))));
        on('replay-export-btn', 'click', () => exclusive(async () => jsonDownload(`FLY_LAB_B_replay_${stamp()}.json`, await transport.request('exportReplay'))));
        on('photo-btn', 'click', () => { view.draw(); const c = document.createElement('canvas'); c.width = view.canvas.width; c.height = view.canvas.height; const x = c.getContext('2d'); x.drawImage(view.canvas, 0, 0); x.drawImage(view.overlay, 0, 0); c.toBlob(b => { if (b)
            download(`FLY_LAB_B_view_${stamp()}.png`, b);
        else
            toast('이미지 저장 실패', true); }, 'image/png'); });
        on('load-state-btn', 'click', () => $('state-file').click());
        on('load-graph-btn', 'click', () => $('graph-file').click());
        async function readJSON(input) { const file = input.files[0]; input.value = ''; if (!file)
            return null; if (file.size > 10 * 1024 * 1024)
            throw Error('B의 가져오기 한도는 파일당 10 MB입니다.'); return JSON.parse(await file.text()); }
        on('state-file', 'change', () => exclusive(async () => { const s = await readJSON($('state-file')); if (!s)
            return; const op = s.schema === 'flylab.checkpoint.v2' ? 'restore' : s.schema === 'flylab.replay.v2' ? 'replay' : s.schema === 'flylab.experiment.v2' ? 'initExperiment' : null; if (!op)
            throw Error('체크포인트 또는 리플레이 파일이 아닙니다.'); const r = await transport.request(op, s); configureReady(r); $('seed').value = String(s.seed ?? s.initial?.seed ?? 42); setPaused(true); toast('파일을 복원했습니다. 재생을 누르면 이어서 계산합니다.'); }));
        on('graph-file', 'change', () => exclusive(async () => { const s = await readJSON($('graph-file')); if (!s)
            return; F.validateGraph(s); s.manifest = { ...(s.manifest || {}), userImport: true, scope: 'user_import_unverified', neurons: s.nodes.length, directedPairs: s.edges.length }; const r = await transport.request('init', { graph: s, seed: Number($('seed').value), config: frame.config }); configureReady(r); toast('사용자 연결망을 적용하고 기록을 초기화했습니다. 출처는 인증되지 않았습니다.'); }));
        on('paired-btn', 'click', () => exclusive(async () => { const wasPaused = paused; setPaused(true); $('paired-btn').disabled = true; try {
            paired = await transport.request('paired', { kind: $('experiment-kind').value, seconds: 4 });
            text('paired-summary', `대조군 대비 평균 경로 차이 ${paired.meanPathSeparation.toFixed(2)} mm`);
            text('paired-title', '같은 시작 상태 · 두 개의 경로');
            const box = $('paired-metrics');
            box.replaceChildren();
            for (const run of paired.runs) {
                const card = document.createElement('div');
                card.className = 'paired-metric';
                const title = document.createElement('strong');
                title.textContent = run.variant === 'control' ? '대조군' : '개입군';
                const p = document.createElement('p');
                p.textContent = `이동 ${run.metrics.distance.toFixed(2)} mm · 방향 오차 ${run.metrics.meanHeadingErrorDeg.toFixed(1)}° · 접촉 ${run.metrics.collisions}`;
                card.append(title, p);
                box.append(card);
            }
            const note = document.createElement('p');
            note.className = 'muted';
            note.textContent = `시뮬레이션 4초, 1초 시점 개입. 평균 경로 차이 ${paired.meanPathSeparation.toFixed(3)} mm. 현재 실험 상태는 변경되지 않습니다. 실제 초파리의 실험 결과가 아닙니다.`;
            box.append(note);
            openDialog('paired-dialog');
        }
        finally {
            $('paired-btn').disabled = false;
            setPaused(wasPaused);
        } }));
        on('paired-export', 'click', () => { if (paired)
            jsonDownload(`FLY_LAB_B_paired_${paired.kind}.json`, paired); });
        on('about-btn', 'click', () => openDialog('about-dialog'));
        on('architecture-btn', 'click', () => openDialog('architecture-dialog'));
        document.querySelectorAll('[data-close]').forEach(b => b.addEventListener('click', () => $(b.dataset.close).close()));
        document.addEventListener('keydown', e => { if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || document.querySelector('dialog[open]'))
            return; if (e.code === 'Space') {
            e.preventDefault();
            setPaused(!paused);
        }
        else if (e.key.toLowerCase() === 'f')
            view.track();
        else if (e.key.toLowerCase() === 'r')
            reset().catch(err => toast(err.message, true));
        else if (e.key === 'Escape') {
            placement = null;
            $('placement-banner').hidden = true;
            document.querySelectorAll('[data-place]').forEach(b => b.classList.remove('active'));
        } });
        document.addEventListener('visibilitychange', () => { accumulator = 0; lastWall = performance.now(); });
        window.addEventListener('resize', drawMonitors);
        window.addEventListener('fly-render-error', () => { text('render-error', '그래픽 컨텍스트가 중단되었습니다. 파일을 다시 열어 주세요.'); $('render-error').hidden = false; });
    }
    function animate(now) {
        rafId = requestAnimationFrame(animate);
        if (!frame)
            return;
        const elapsed = Math.min((now - lastWall) / 1000, .1);
        lastWall = now;
        if (document.hidden) {
            accumulator = 0;
            return;
        }
        if (!paused && !locked) {
            accumulator = Math.min(accumulator + elapsed * speed, .3 * speed);
            if (!busy) {
                const n = Math.min(10, Math.floor(accumulator / F.DT));
                if (n > 0) {
                    busy = true;
                    accumulator -= n * F.DT;
                    transport.request('advance', { steps: n }).then(accept).catch(e => { setPaused(true); toast(e.message, true); }).finally(() => busy = false);
                }
            }
        }
        view.draw();
        fpsFrames++;
        if (now - fpsStart > 1000) {
            fps = fpsFrames * 1000 / (now - fpsStart);
            fpsFrames = 0;
            fpsStart = now;
            text('runtime', `${transport.name} · ${view.engine} · ${fps.toFixed(0)} FPS · 신경 5 ms · 물리 0.1 ms`);
        }
        if (now - lastHUD > 100) {
            drawMonitors();
            lastHUD = now;
        }
    }
    async function start() { try {
        view = new F.WorldView($('world'), $('world-overlay'));
        text('backend-state','로컬 물리 서버 연결 중…');
        transport = new F.BTransport();
        await transport.connect();
        text('backend-state','모델 구성·접촉 안정화 중…');
        const ready=await transport.request('init',{graph:baseGraph,seed:42});
        $('backend-gate').hidden=true;
        configureReady(ready);
        initControls();
        setPaused(false);
        lastWall = performance.now();
        requestAnimationFrame(animate);
        window.flyLab.ready = true;
        return ready;
    }
    catch (err) {
        toast('시작 실패: ' + err.message, true);text('backend-state',err.message);$('backend-gate').hidden=false;
        $('render-error').hidden = false;
        text('render-error', '실행할 수 없습니다: ' + err.message);
        throw err;
    } }
    window.flyLab = { ready: false, get frame() { return frame; }, get catalog() { return catalog; }, get transport() { return transport; }, get view() { return view; }, get selected() { return selected; }, get paused() { return paused; }, get paired() { return paired; }, command, choose, setPaused, reset, async advanceSteps(n) { return exclusive(async () => { setPaused(true); while (n > 0) {
            const m = Math.min(n, 10);
            accept(await transport.request('advance', { steps: m }));
            n -= m;
        } view.draw(); drawMonitors(); return frame; }); }, async checkpoint() { return exclusive(() => transport.request('checkpoint')); }, async restore(s) { return exclusive(async () => { configureReady(await transport.request('restore', s)); setPaused(true); return frame; }); }, dispose() { cancelAnimationFrame(rafId); transport?.close(); view?.dispose(); } };
    window.flyLab.readyPromise = start(); window.flyLab.readyPromise.catch(()=>{});
})(globalThis.Fly);
