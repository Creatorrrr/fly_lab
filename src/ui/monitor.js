(function (F) {
    'use strict';
    const palette = ['#7de2c4', '#e9bf7e', '#91b7ef', '#ed9789', '#d1b8e0', '#78c9dc', '#c6db94', '#ecb0c6'];
    function setup(canvas) { const r = canvas.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2); if (!r.width || !r.height)
        return null; const w = Math.round(r.width * d), h = Math.round(r.height * d); if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    } const ctx = canvas.getContext('2d'); ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0, 0, r.width, r.height); return { ctx, w: r.width, h: r.height }; }
    const heat = a => { a = F.clamp(a); const lo = [19, 43, 52], hi = [149, 235, 187], m = [34, 133, 113]; const c = a < .55 ? lo.map((x, i) => F.mix(x, m[i], a / .55)) : m.map((x, i) => F.mix(x, hi[i], (a - .55) / .45)); return 'rgb(' + c.map(Math.round).join(',') + ')'; };
    function compass(canvas, f) {
        const d = setup(canvas);
        if (!d)
            return;
        const { ctx, w, h } = d, cx = w / 2, cy = h / 2, r = Math.min(w, h) * .40;
        ctx.strokeStyle = '#29424a';
        ctx.lineWidth = 1;
        for (const radius of [r, r * .72]) {
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, F.TAU);
            ctx.stroke();
        }
        ctx.font = '8px ui-monospace,monospace';
        ctx.fillStyle = '#779599';
        ctx.textAlign = 'center';
        for (let k = 0; k < 8; k++) {
            const a = k * F.TAU / 8, x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(cx + Math.cos(a) * (r - 4), cy + Math.sin(a) * (r - 4));
            ctx.stroke();
        }
        ctx.fillText('0°', cx + r + 9, cy + 3);
        ctx.fillText('90°', cx, cy + r + 10);
        const bins = f.neural.headingBins;
        for (let k = 0; k < 8; k++) {
            const a = k / 8 * F.TAU;
            ctx.strokeStyle = heat(bins[k]);
            ctx.lineWidth = 5;
            ctx.beginPath();
            ctx.arc(cx, cy, r * .73, a - .25, a + .25);
            ctx.stroke();
        }
        const arrow = (a, color, len, dash = false) => { ctx.strokeStyle = color; ctx.lineWidth = dash ? 1 : 1.8; ctx.setLineDash(dash ? [3, 3] : []); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a) * r * len, cy + Math.sin(a) * r * len); ctx.stroke(); ctx.setLineDash([]); if (!dash) {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(cx + Math.cos(a) * r * len, cy + Math.sin(a) * r * len, 2.3, 0, F.TAU);
            ctx.fill();
        } };
        arrow(f.body.yaw, '#829aaf', 1, true);
        arrow(f.neural.goal, '#e9bf7e', .91);
        arrow(f.neural.heading, '#7de2c4', .94);
        ctx.fillStyle = '#d8e6dc';
        ctx.beginPath();
        ctx.arc(cx, cy, 2, 0, F.TAU);
        ctx.fill();
    }
    function sensors(canvas, f) { const d = setup(canvas); if (!d)
        return; const { ctx, w, h } = d, s = f.sensors; if (!s)
        return; ctx.fillStyle = '#729499'; ctx.font = '8px ui-monospace,monospace'; ctx.fillText('CUE', 0, 10); ctx.fillText('NEAR', 0, 44); const pad = 37, width = w - pad; for (let i = 0; i < 64; i++) {
        ctx.fillStyle = heat(s.panorama[i]);
        ctx.fillRect(pad + i * width / 64, 0, width / 64 - .6, 20);
    } for (let i = 0; i < 9; i++) {
        const v = 1 - s.nearRanges[i] / 10;
        ctx.fillStyle = heat(v);
        ctx.fillRect(pad + i * width / 9, 31, width / 9 - 3, 20);
    } ctx.strokeStyle = '#3d5960'; ctx.beginPath(); ctx.moveTo(pad + width / 2, 22); ctx.lineTo(pad + width / 2, 27); ctx.stroke(); }
    class NeuralMonitor {
        constructor(catalog, graph) { this.setCatalog(catalog, graph); this.hits = []; this.cells = []; }
        setCatalog(catalog, graph) { this.catalog = catalog; this.graph = graph; this.byId = Object.fromEntries(catalog.map(n => [n.id, n])); }
        network(canvas, f, selected) {
            const d = setup(canvas);
            if (!d)
                return;
            const { ctx, w, h } = d, vals = Object.fromEntries(f.neural.ids.map((id, i) => [id, f.neural.values[i]]));
            const layout = { EPG: [.38, .45, .255], PENa: [.79, .24, .105], PENb: [.79, .76, .105], PEG: [.13, .78, .090] }, pos = {};
            this.hits = [];
            for (const [type, [cx, cy, rr]] of Object.entries(layout)) {
                const ns = this.catalog.filter(n => n.type === type);
                const r = Math.min(w, h) * rr;
                ns.forEach((n, i) => { const a = i / ns.length * F.TAU, p = { id: n.id, x: cx * w + r * Math.cos(a), y: cy * h + r * Math.sin(a) }; pos[n.id] = p; this.hits.push(p); });
                ctx.fillStyle = '#8faeac';
                ctx.font = '9px ui-monospace,monospace';
                ctx.textAlign = 'center';
                ctx.fillText(type, cx * w, cy * h + 3);
                ctx.fillStyle = '#4e7177';
                ctx.font = '7px ui-monospace,monospace';
                ctx.fillText(ns.length + ' IDs', cx * w, cy * h + 14);
            }
            for (const e of this.graph.edges) {
                const a = pos[e.pre], b = pos[e.post];
                if (!a || !b)
                    continue;
                const relevant = e.pre === selected || e.post === selected;
                ctx.strokeStyle = relevant ? '#7fe4c978' : '#39796b13';
                ctx.lineWidth = relevant ? Math.min(1.6, .5 + e.count / 70) : .5;
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
                ctx.quadraticCurveTo(mx, my - 7, b.x, b.y);
                ctx.stroke();
            }
            const lesions = new Set(f.neural.lesions);
            for (const p of this.hits) {
                ctx.fillStyle = lesions.has(p.id) ? '#a76e68' : heat(vals[p.id] || 0);
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.id === selected ? 4 : 2.2, 0, F.TAU);
                ctx.fill();
                if (p.id === selected) {
                    ctx.strokeStyle = '#daf9e5';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
            ctx.textAlign = 'left';
            ctx.font = '7px ui-monospace,monospace';
            ctx.fillStyle = '#527c7a';
            ctx.fillText('EB / PB · ' + this.graph.edges.length + ' PAIRS', 4, 10);
            if (selected.startsWith('model:')) {
                ctx.fillStyle = '#d7b884';
                ctx.fillText('선택: 설계 노드 (아래 활성도 맵)', w * .33, h - 3);
            }
        }
        heatmap(canvas, f, selected, filtered) {
            const d = setup(canvas);
            if (!d)
                return;
            const { ctx, w, h } = d, cols = 20, rows = Math.max(1, Math.ceil(filtered.length / cols)), gap = 2, cw = (w - gap * (cols - 1)) / cols, ch = Math.min(17, (h - gap * (rows - 1)) / rows), vals = Object.fromEntries(f.neural.ids.map((id, i) => [id, f.neural.values[i]])), lesions = new Set(f.neural.lesions);
            this.cells = [];
            filtered.forEach((n, i) => { const x = (i % cols) * (cw + gap), y = Math.floor(i / cols) * (ch + gap); ctx.fillStyle = heat(vals[n.id] || 0); ctx.fillRect(x, y, cw, ch); if (n.origin === 'model') {
                ctx.fillStyle = '#c2a16c';
                ctx.fillRect(x, y + ch - 1, cw, 1);
            } if (lesions.has(n.id)) {
                ctx.strokeStyle = '#e98c7c';
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + cw, y + ch);
                ctx.stroke();
            } if (n.id === selected) {
                ctx.strokeStyle = '#e4f7dc';
                ctx.lineWidth = 1.2;
                ctx.strokeRect(x + .5, y + .5, cw - 1, ch - 1);
            } this.cells.push({ id: n.id, x, y, w: cw, h: ch }); });
            if (!filtered.length) {
                ctx.fillStyle = '#7f9da0';
                ctx.font = '11px sans-serif';
                ctx.fillText('일치하는 노드가 없습니다.', 8, 27);
            }
        }
        pickNetwork(x, y) { let best = null, d = 10; for (const p of this.hits) {
            const k = Math.hypot(x - p.x, y - p.y);
            if (k < d) {
                best = p.id;
                d = k;
            }
        } return best; }
        pickHeat(x, y) { return this.cells.find(p => x >= p.x && x <= p.x + p.w && y >= p.y && y <= p.y + p.h)?.id; }
    }
    function timeline(canvas, history, pins, catalog, f) {
        const d = setup(canvas);
        if (!d)
            return;
        const { ctx, w, h } = d, pad = 29, right = 8, top = 10, bottom = 23, end = Math.max(12, f.simTime), start = Math.max(0, end - 120), iw = w - pad - right, ih = h - top - bottom;
        ctx.font = '8px ui-monospace,monospace';
        ctx.textAlign = 'right';
        for (const v of [0, .25, .5, .75, 1]) {
            const y = top + (1 - v) * ih;
            ctx.strokeStyle = '#20343b';
            ctx.beginPath();
            ctx.moveTo(pad, y);
            ctx.lineTo(w - right, y);
            ctx.stroke();
            ctx.fillStyle = '#729197';
            ctx.fillText(v.toFixed(2), pad - 6, y + 3);
        }
        ctx.textAlign = 'center';
        for (let i = 0; i <= 6; i++) {
            const t = start + i * (end - start) / 6, x = pad + i * iw / 6;
            ctx.fillStyle = '#6f8d95';
            ctx.fillText(t.toFixed(0) + 's', x, h - 5);
        }
        pins.forEach((id, k) => { ctx.strokeStyle = palette[k % palette.length]; ctx.lineWidth = 1.5; ctx.beginPath(); let moved = false; for (const row of history) {
            if (row.t < start)
                continue;
            const value = row.values[id];
            if (value === undefined)
                continue;
            const x = pad + (row.t - start) / (end - start) * iw, y = top + (1 - value) * ih;
            if (!moved) {
                ctx.moveTo(x, y);
                moved = true;
            }
            else
                ctx.lineTo(x, y);
        } ctx.stroke(); });
    }
    function pairedChart(canvas, r) { const d = setup(canvas); if (!d)
        return; const { ctx, w, h } = d, pad = 33, iw = w - pad * 2, ih = h - pad * 2; ctx.font = '9px ui-monospace,monospace'; ctx.fillStyle = '#819da2'; ctx.textAlign = 'center'; const xy = p => [pad + (p[0] + 24) / 48 * iw, pad + (18 - p[2]) / 36 * ih]; for (let x = -24; x <= 24; x += 8) {
        const [a, b] = xy([x, 0, -18]), [c, e] = xy([x, 0, 18]);
        ctx.strokeStyle = '#223b43';
        ctx.beginPath();
        ctx.moveTo(a, b);
        ctx.lineTo(c, e);
        ctx.stroke();
        ctx.fillText(String(x), a, h - 15);
    } for (let z = -18; z <= 18; z += 6) {
        const [a, b] = xy([-24, 0, z]), [c, e] = xy([24, 0, z]);
        ctx.beginPath();
        ctx.moveTo(a, b);
        ctx.lineTo(c, e);
        ctx.stroke();
        ctx.fillText(String(z), 14, b + 3);
    } r.runs.forEach((run, k) => { ctx.strokeStyle = palette[k]; ctx.lineWidth = 2; ctx.beginPath(); run.trace.forEach((s, i) => { const [x, y] = xy(s.p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); const [x, y] = xy(run.trace.at(-1).p); ctx.fillStyle = palette[k]; ctx.beginPath(); ctx.arc(x, y, 3, 0, F.TAU); ctx.fill(); ctx.textAlign = 'left'; ctx.fillText(k ? '개입군' : '대조군', w - 120, 16 + k * 15); }); ctx.textAlign = 'center'; ctx.fillStyle = '#819da2'; ctx.fillText('x (mm)', w / 2, h - 1); }
    F.Monitor = { NeuralMonitor, compass, sensors, timeline, pairedChart, heat, palette };
})(globalThis.Fly);
