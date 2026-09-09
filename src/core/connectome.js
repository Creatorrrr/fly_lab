(function (F) {
    'use strict';
    // A uses receiver-normalized sqrt(synapse count), not an anatomical conductance.
    function validateGraph(g) {
        if (!g || g.schema !== 'flylab.connectome.v1' || !Array.isArray(g.nodes) || !Array.isArray(g.edges))
            throw Error('연결망 형식이 flylab.connectome.v1이 아닙니다.');
        if (g.nodes.length < 1 || g.nodes.length > 440 || g.edges.length > 100000)
            throw Error('A 로더 한도: 실제 ID 440개 / 연결 100,000개 (보조 노드 포함 512개). C는 별도 계산기가 필요합니다.');
        const ids = new Set();
        for (const n of g.nodes) {
            if (typeof n.id !== 'string' || n.id.length > 160 || ids.has(n.id) || !['EPG', 'PENa', 'PENb', 'PEG'].includes(n.type))
                throw Error('노드 ID/형식 오류. A는 EPG/PENa/PENb/PEG를 지원합니다.');
            if (!/^hemibrain:[0-9]+$/.test(n.id) || typeof n.bodyId !== 'string' || !/^\d+$/.test(n.bodyId) || n.id !== 'hemibrain:' + n.bodyId || typeof n.name !== 'string' || n.name.length > 200 || !['L', 'R'].includes(n.hemisphere) || !Number.isInteger(n.pbIndex) || n.pbIndex < 1 || n.pbIndex > 9)
                throw Error('문자열 bodyId/hemibrain ID/좌우 분류 오류');
            ids.add(n.id);
            if (!Number.isInteger(n.indexFix) || n.indexFix < 1 || n.indexFix > 9)
                throw Error('indexFix must be 1..9');
            if (n.origin !== 'anatomical')
                throw Error('해부학적 노드 origin 오류');
        }
        if (!g.nodes.some(n => n.type === 'EPG'))
            throw Error('A 제어기에는 EPG 노드가 필요합니다.');
        const seen = new Set();
        for (const e of g.edges) {
            if (!ids.has(e.pre) || !ids.has(e.post) || !Number.isFinite(e.count) || e.count <= 0)
                throw Error('연결 대상/연결량 오류');
            const k = e.pre + '|' + e.post;
            if (seen.has(k))
                throw Error('동일 뉴런 쌍의 ROI 연결은 먼저 합산해 주세요.');
            seen.add(k);
        }
        return g;
    }
    class Circuit {
        constructor(data) {
            this.data = F.clone(validateGraph(data));
            this.nodes = this.data.nodes.map(n => ({ ...n, angle: ((n.indexFix - 1) % 8) * F.TAU / 8, displayName: n.type + ' ' + n.hemisphere + n.pbIndex + ' · ' + n.bodyId.slice(-4) }));
            this.realCount = this.nodes.length;
            this.pop = {};
            this.index = Object.create(null);
            const add = (type, count, extra = {}) => { for (let k = 0; k < count; k++)
                this.nodes.push({ id: 'model:' + type + ':' + k, name: type + ' ' + String(k + 1).padStart(2, '0'), displayName: type + ' ' + String(k + 1).padStart(2, '0'), type, origin: 'model', angle: k / count * F.TAU, ...extra }); };
            add('HD', 16);
            add('GOAL', 16);
            add('PFL3-L', 16);
            add('PFL3-R', 16);
            add('DN', 2);
            add('SENSE', 4);
            add('ALT', 2);
            this.nodes.forEach((n, i) => { this.index[n.id] = i; (this.pop[n.type] ??= []).push(i); });
            this.N = this.nodes.length;
            const incoming = Array.from({ length: this.N }, () => []);
            for (const e of data.edges) {
                incoming[this.index[e.post]].push([this.index[e.pre], Math.sqrt(e.count), e.count]);
            }
            this.rowPtr = new Uint32Array(this.N + 1);
            const source = [], weight = [];
            incoming.forEach((row, i) => { const sum = row.reduce((s, e) => s + e[1], 0) || 1; for (const e of row) {
                source.push(e[0]);
                weight.push(e[1] / sum);
            } this.rowPtr[i + 1] = source.length; });
            this.source = new Uint32Array(source);
            this.weight = new Float64Array(weight);
        }
        propagate(a, gain = 1) { const r = new Float64Array(this.N); for (let j = 0; j < this.realCount; j++)
            for (let k = this.rowPtr[j]; k < this.rowPtr[j + 1]; k++)
                r[j] += this.weight[k] * a[this.source[k]] * gain; return r; }
        catalog() { return this.nodes.map((n, i) => ({ ...n, index: i })); }
    }
    F.validateGraph = validateGraph;
    F.Circuit = Circuit;
})(globalThis.Fly);
