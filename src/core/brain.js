(function (F) {
    'use strict';
    const { clamp, wrap, TAU } = F;
    /* Hybrid demonstrator, NOT a fitted biological circuit or a connectome-only
     * emulation. All state origins and controller assumptions are disclosed.
     * Critically: step() has no access to body/world ground truth. */
    class NeuralController {
        constructor(circuit, seed) { this.circuit = circuit; this.rng = new F.RNG(seed ^ 0x7cab1); this.a = new Float64Array(circuit.N); this.a.fill(.05); this.adapt = new Float64Array(circuit.N); this.lesions = new Set(); this.stim = new Map(); this.phaseEstimate = 0; this.heading = 0; this.confidence = 0; this.goal = 1.0; this.clock = 0; this.nextGoal = 9; this.avoidSide = 1; this.prevContact = 0; this.odorMode = 'explore'; this.turnMemory = 0; this.contactTime = 0; this.recoveryTime = 0; this.output = { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0 }; this.lastCue = null; this.error = 0; this.recNorm = 0; }
        val(type, k = 0) { return this.a[this.circuit.pop[type][k]] || 0; }
        mean(type) { let is = this.circuit.pop[type]; return is.reduce((s, i) => s + this.a[i], 0) / is.length; }
        setTarget(i, value, dt, tau = .13) { if (this.lesions.has(i)) {
            this.a[i] = 0;
            return;
        } const stim = this.stim.get(i); const extra = stim ? stim.amplitude : 0; this.a[i] += (clamp(value + extra) - this.a[i]) * (1 - Math.exp(-dt / tau)); }
        step(s, dt, config) {
            this.clock += dt;
            const c = this.circuit, old = this.a.slice();
            // Decoding of the far-field cue occurs inside the brain from 64 observations.
            let cx = 0, cy = 0, cs = 0;
            for (let i = 0; i < 64; i++) {
                const p = i / 64 * TAU - Math.PI;
                cx += s.panorama[i] * Math.cos(p);
                cy += s.panorama[i] * Math.sin(p);
                cs += s.panorama[i];
            }
            const visible = cs > .8;
            this.lastCue = visible ? wrap(-Math.atan2(cy, cx)) : null;
            // Explicit angular-observer scaffold (HD); no direct assignment from true yaw.
            this.phaseEstimate = wrap(this.phaseEstimate + s.angularVelocity * dt);
            if (visible)
                this.phaseEstimate = wrap(this.phaseEstimate + wrap(this.lastCue - this.phaseEstimate) * (1 - Math.exp(-dt / .24)));
            for (const i of c.pop.HD)
                this.setTarget(i, .05 + .90 * Math.exp(3.4 * (Math.cos(c.nodes[i].angle - this.phaseEstimate) - 1)), dt, .07);
            let hx = 0, hy = 0;
            for (const i of c.pop.HD) {
                const a = Math.max(0, this.a[i] - .04);
                hx += a * Math.cos(c.nodes[i].angle);
                hy += a * Math.sin(c.nodes[i].angle);
            }
            const scaffold = Math.atan2(hy, hx);
            const rec = c.propagate(old, config.graphGain);
            this.recNorm = rec.reduce((x, y) => x + y, 0) / c.realCount;
            for (let i = 0; i < c.realCount; i++) {
                const n = c.nodes[i], bump = Math.exp(3 * (Math.cos(n.angle - scaffold) - 1));
                const omega = clamp(s.angularVelocity / 2, -1, 1);
                let target = .055 + .77 * bump;
                if (n.type === 'PENa' || n.type === 'PENb')
                    target = .07 + .58 * bump + .18 * Math.max(0, omega * (n.hemisphere === 'R' ? 1 : -1));
                if (n.type === 'PEG')
                    target = .07 + .66 * bump;
                target += .30 * rec[i] - .045;
                this.adapt[i] += (old[i] - this.adapt[i]) * dt / 2;
                target -= .025 * this.adapt[i];
                this.setTarget(i, target, dt, n.type === 'PEG' ? .28 : .11);
            }
            // Equalize cell counts by anatomical column before decoding EPG heading.
            const bins = new Float64Array(8), counts = new Uint16Array(8);
            for (const i of c.pop.EPG) {
                const b = (c.nodes[i].indexFix - 1) % 8;
                bins[b] += Math.max(0, this.a[i] - .025);
                counts[b]++;
            }
            let ex = 0, ey = 0, total = 0;
            for (let k = 0; k < 8; k++) {
                bins[k] /= Math.max(1, counts[k]);
                ex += bins[k] * Math.cos(k * TAU / 8);
                ey += bins[k] * Math.sin(k * TAU / 8);
                total += bins[k];
            }
            this.confidence = total > .01 ? Math.hypot(ex, ey) / total : 0;
            if (total > .01)
                this.heading = Math.atan2(ey, ex);
            this.headingBins = Array.from(bins);
            if (!visible && this.confidence > .1)
                this.phaseEstimate = wrap(this.phaseEstimate + wrap(this.heading - this.phaseEstimate) * dt * .08);
            let dangerL = Math.max(...s.nearRanges.slice(0, 4).map(r => Math.exp(-r / 2.2))), dangerR = Math.max(...s.nearRanges.slice(5).map(r => Math.exp(-r / 2.2))), front = Math.exp(-s.nearRanges[4] / 2.2);
            const clearance = Math.min(...s.nearRanges.slice(3, 6));
            if (this.turnMemory) {
                const direction = this.turnMemory > 0 ? 1 : -1;
                const remaining = clamp(Math.abs(this.turnMemory) - direction * s.angularVelocity * dt, .001, TAU);
                this.turnMemory = direction * remaining;
                if (remaining <= .05 && clearance > 4.5 && !s.contact) this.turnMemory = 0;
            } else if (clearance < 3.5 || s.contact) {
                if (Math.abs(dangerL - dangerR) > .08) this.avoidSide = dangerL > dangerR ? 1 : -1;
                this.turnMemory = this.avoidSide * 1.3;
            }
            this.prevContact = s.contact;
            const avoiding = !!this.turnMemory;
            this.recoveryTime = Math.max(0, this.recoveryTime - dt);
            this.contactTime = clamp(this.contactTime + (s.contact && avoiding ? dt : -dt * .5), 0, 1);
            if (avoiding && !this.recoveryTime && this.contactTime > .3) {
                this.recoveryTime = .4;
                this.contactTime = 0;
            }
            this.setTarget(c.pop.SENSE[0], dangerL, dt, .08);
            this.setTarget(c.pop.SENSE[1], dangerR, dt, .08);
            this.setTarget(c.pop.SENSE[2], s.odor[0], dt, .15);
            this.setTarget(c.pop.SENSE[3], s.odor[1], dt, .15);
            if (config.task === 'heading') {
                this.goal = config.goalAngle;
                this.odorMode = 'heading';
            }
            else {
                if (this.clock > this.nextGoal) {
                    this.goal = wrap(this.goal + (.5 + this.rng.next()) * this.rng.signed() * 2);
                    this.nextGoal = this.clock + 7 + this.rng.next() * 7;
                }
                if (Math.max(...s.odor) > .10) {
                    const odorDiff = this.val('SENSE', 3) - this.val('SENSE', 2);
                    this.goal = wrap(this.heading + clamp(odorDiff * 9, -1, 1) + (s.odorChange < -.02 ? .50 : 0));
                    this.odorMode = 'chemotaxis';
                }
                else
                    this.odorMode = 'explore';
                if (Math.max(...s.odor) > .88) {
                    this.goal = wrap(this.heading + .7);
                    this.odorMode = 'dwell';
                }
            }
            for (const i of c.pop.GOAL)
                this.setTarget(i, .02 + .94 * Math.exp(3.3 * (Math.cos(c.nodes[i].angle - this.goal) - 1)), dt, .2);
            let gx = 0, gy = 0;
            for (const i of c.pop.GOAL) {
                gx += this.a[i] * Math.cos(c.nodes[i].angle);
                gy += this.a[i] * Math.sin(c.nodes[i].angle);
            }
            const goalRead = Math.atan2(gy, gx);
            this.error = wrap(goalRead - this.heading);
            let avoid = clamp(this.val('SENSE', 0) - this.val('SENSE', 1) + Math.max(0, s.danger - .25) * this.avoidSide, -1, 1), e = this.error;
            if (avoiding) avoid = .85 * (this.turnMemory > 0 ? 1 : -1);
            // PFL3-inspired *synthetic* directional readout; not claimed as anatomical PFL3.
            for (let k = 0; k < 16; k++) {
                const attention = avoiding ? 1 : .30 + .70 * this.a[c.pop.GOAL[k]], base = .045;
                const left = base + attention * (.68 * (!avoiding) * Math.max(0, -Math.sin(e)) + .42 * Math.max(0, -avoid) * 3 + .22 * s.contact * (this.avoidSide < 0));
                const right = base + attention * (.68 * (!avoiding) * Math.max(0, Math.sin(e)) + .42 * Math.max(0, avoid) * 3 + .22 * s.contact * (this.avoidSide > 0));
                this.setTarget(c.pop['PFL3-L'][k], left, dt, .10);
                this.setTarget(c.pop['PFL3-R'][k], right, dt, .10);
            }
            this.setTarget(c.pop.DN[0], clamp(this.mean('PFL3-L') * 2.6), dt, .09);
            this.setTarget(c.pop.DN[1], clamp(this.mean('PFL3-R') * 2.6), dt, .09);
            const heightError = config.altitude - s.clearanceDown;
            this.setTarget(c.pop.ALT[0], clamp(.5 + heightError * .2), dt, .25);
            this.setTarget(c.pop.ALT[1], clamp(.5 - heightError * .2), dt, .25);
            const drive = clamp(this.mean('EPG') * 4, 0, 1);
            let speed = (config.mode === 'walk' ? 3.3 : 5.3) * drive * (1 - clamp(front * .72, 0, .8));
            if (this.odorMode === 'dwell')
                speed *= .4;
            if (s.danger > .3)
                speed *= 1.15;
            if (avoiding) speed = this.recoveryTime ? -2.5 : 0;
            // All yaw control passes through DN activity. No hidden geometric steering.
            this.output = { yawRate: config.motorCoupled ? clamp(3.5 * (this.val('DN', 1) - this.val('DN', 0)), -3, 3) : 0,
                forwardSpeed: config.motorCoupled ? speed : 0, verticalSpeed: config.motorCoupled && config.mode === 'flight' ? 2.7 * (this.val('ALT', 0) - this.val('ALT', 1)) : 0 };
            for (const [i, x] of this.stim) {
                x.remaining -= dt;
                if (x.remaining <= 1e-10)
                    this.stim.delete(i);
            }
            return this.output;
        }
        intervene(ids, kind, amplitude = .65, duration = 2) { for (const id of ids) {
            const i = this.circuit.index[id];
            if (i === undefined)
                throw Error('알 수 없는 뉴런 ' + id);
            if (kind === 'stimulate')
                this.stim.set(i, { amplitude, remaining: duration });
            else if (kind === 'suppress') {
                this.lesions.add(i);
                this.a[i] = 0;
            }
            else if (kind === 'release')
                this.lesions.delete(i);
        } }
        snapshot() { return { a: Array.from(this.a), adapt: Array.from(this.adapt), lesions: [...this.lesions], stim: [...this.stim], rng: this.rng.state, phaseEstimate: this.phaseEstimate, heading: this.heading, confidence: this.confidence, goal: this.goal, clock: this.clock, nextGoal: this.nextGoal, avoidSide: this.avoidSide, prevContact: this.prevContact, odorMode: this.odorMode, turnMemory: this.turnMemory, contactTime: this.contactTime, recoveryTime: this.recoveryTime, output: { ...this.output }, lastCue: this.lastCue, error: this.error, recNorm: this.recNorm, headingBins: this.headingBins || Array(8).fill(0) }; }
        restore(s) {
            const keys = Object.keys(this.snapshot()).sort().join('|');
            if (!s || typeof s !== 'object' || Array.isArray(s) || Object.keys(s).sort().join('|') !== keys)
                throw Error('Unknown neural state fields');
            const num = (v, name, lo = -1e12, hi = 1e12) => {
                if (typeof v !== 'number' || !Number.isFinite(v) || v < lo || v > hi)
                    throw Error('Invalid neural state: ' + name);
            };
            const idx = v => { num(v, 'index', 0, this.circuit.N - 1); if (!Number.isInteger(v)) throw Error('Integer index required'); };
            for (const k of ['a', 'adapt']) {
                if (!Array.isArray(s[k]) || s[k].length !== this.circuit.N) throw Error('Invalid neural shape');
                s[k].forEach(v => num(v, k, 0, 1));
            }
            if (!['explore','heading','chemotaxis','dwell'].includes(s.odorMode)) throw Error('Invalid neural mode');
            if (!s.output || Object.keys(s.output).sort().join('|') !== 'forwardSpeed|verticalSpeed|yawRate') throw Error('Invalid output');
            Object.values(s.output).forEach(v => num(v,'motor',-20,20));
            if (s.lastCue !== null) num(s.lastCue,'lastCue',-Math.PI,Math.PI);
            if (!Array.isArray(s.headingBins) || s.headingBins.length !== 8) throw Error('Invalid heading bins');
            s.headingBins.forEach(v => num(v,'headingBin',0,1));
            if (!Array.isArray(s.lesions) || !Array.isArray(s.stim)) throw Error('Invalid intervention state');
            s.lesions.forEach(idx);
            if (new Set(s.lesions).size !== s.lesions.length) throw Error('Duplicate lesion');
            const used = new Set();
            for (const pair of s.stim) {
                if (!Array.isArray(pair) || pair.length !== 2) throw Error('Invalid stimulus pair');
                const [i,v] = pair; idx(i);
                if (used.has(i)) throw Error('Duplicate stimulus'); used.add(i);
                if (!v || Object.keys(v).sort().join('|') !== 'amplitude|remaining') throw Error('Invalid stimulus');
                num(v.amplitude,'amplitude',-1,1); num(v.remaining,'remaining',1e-12,60);
            }
            num(s.rng,'rng',1,4294967295); if (!Number.isInteger(s.rng)) throw Error('Integer RNG required');
            for (const [k,lo,hi] of [
                ['phaseEstimate',-Math.PI,Math.PI],['heading',-Math.PI,Math.PI],
                ['confidence',0,1+1e-12],['goal',-Math.PI,Math.PI],['clock',0,1e12],['nextGoal',0,1e12],
                ['turnMemory',-2*Math.PI,2*Math.PI],['contactTime',0,1],['recoveryTime',0,.4+1e-12],
                ['error',-Math.PI,Math.PI],['recNorm',0,2+1e-12]]) num(s[k],k,lo,hi);
            if (![-1,1].includes(s.avoidSide) || ![0,1].includes(s.prevContact)) throw Error('Invalid discrete state');
            const { a, adapt, lesions, stim, rng, ...rest } = s;
            this.a = Float64Array.from(a); this.adapt = Float64Array.from(adapt);
            this.lesions = new Set(lesions); this.stim = new Map(F.clone(stim));
            this.rng.state = rng >>> 0; Object.assign(this, F.clone(rest));
        }
    }
    F.NeuralController = NeuralController;
})(globalThis.Fly);
