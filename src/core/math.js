/* FLY LAB A — dependency-free core. Also runs unchanged in Node and Workers. */
(function (root) {
    'use strict';
    const F = root.Fly = root.Fly || {};
    F.VERSION = '0.1.0';
    F.PROTOCOL = 'flylab.protocol.v1';
    F.DT = 1 / 120;
    F.clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
    F.wrap = x => Math.atan2(Math.sin(x), Math.cos(x));
    F.mix = (a, b, t) => a + (b - a) * t;
    F.TAU = 2 * Math.PI;
    F.finite = (v, name = 'number') => { if (typeof v !== 'number' || !Number.isFinite(v))
        throw Error(name + ' must be finite'); return v; };
    F.clone = x => JSON.parse(JSON.stringify(x));
    F.V = { add: (a, b) => a.map((x, i) => x + b[i]), sub: (a, b) => a.map((x, i) => x - b[i]), mul: (a, s) => a.map(x => x * s), dot: (a, b) => a.reduce((s, x, i) => s + x * b[i], 0), cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], len: a => Math.hypot(...a), norm: a => { let l = Math.hypot(...a); return l > 1e-10 ? a.map(x => x / l) : [1, 0, 0]; }, lerp: (a, b, t) => a.map((x, i) => F.mix(x, b[i], t)) };
    class RNG {
        constructor(seed = 42) { this.state = (seed >>> 0) || 1; }
        next() { let x = this.state; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.state = x >>> 0; return this.state / 4294967296; }
        signed() { return 2 * this.next() - 1; }
    }
    F.RNG = RNG;
    class RingBuffer {
        constructor(cap) { this.cap = cap; this.items = new Array(cap); this.cursor = 0; this.length = 0; }
        push(v) { this.items[this.cursor] = v; this.cursor = (this.cursor + 1) % this.cap; this.length = Math.min(this.cap, this.length + 1); }
        array() { let r = []; for (let i = 0; i < this.length; i++)
            r.push(this.items[(this.cursor - this.length + i + this.cap) % this.cap]); return r; }
        clear() { this.cursor = 0; this.length = 0; this.items.fill(undefined); }
    }
    F.RingBuffer = RingBuffer;
})(globalThis);
