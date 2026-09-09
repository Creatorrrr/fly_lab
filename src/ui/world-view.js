(function (F) {
    'use strict';
    const { V, clamp, TAU } = F;
    const M = { mul: (a, b) => { let o = new Float32Array(16); for (let c = 0; c < 4; c++)
            for (let r = 0; r < 4; r++)
                for (let k = 0; k < 4; k++)
                    o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k]; return o; }, perspective: (f, a, n, z) => { let t = 1 / Math.tan(f / 2), m = new Float32Array(16); m[0] = t / a; m[5] = t; m[10] = (z + n) / (n - z); m[11] = -1; m[14] = 2 * z * n / (n - z); return m; }, look: (eye, target) => { const z = V.norm(V.sub(eye, target)), x = V.norm(V.cross([0, 1, 0], z)), y = V.cross(z, x); return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -V.dot(x, eye), -V.dot(y, eye), -V.dot(z, eye), 1]); }, transform: (m, p) => { let v = [...p, 1], r = [0, 0, 0, 0]; for (let i = 0; i < 4; i++)
            for (let j = 0; j < 4; j++)
                r[i] += m[j * 4 + i] * v[j]; return r; } };
    function rgba(c, k = 1) { return `rgba(${c.slice(0, 3).map(x => Math.round(clamp(x * k) * 255)).join(',')},${c[3] ?? 1})`; }
    class Geometry {
        constructor() { this.triangles = []; this.lines = []; }
        tri(a, b, c, color, normal = null) { const n = normal || V.norm(V.cross(V.sub(b, a), V.sub(c, a))); this.triangles.push({ p: [a, b, c], n, c: color }); }
        line(a, b, c) { this.lines.push({ p: [a, b], c }); }
        sphere(p, r, c, transform = x => x, U = 14, W = 8) { const at = (i, j) => { let v = i / W * Math.PI, u = j / U * TAU; return transform([p[0] + r[0] * Math.sin(v) * Math.cos(u), p[1] + r[1] * Math.cos(v), p[2] + r[2] * Math.sin(v) * Math.sin(u)]); }; const center = transform(p); for (let i = 0; i < W; i++)
            for (let j = 0; j < U; j++) {
                let a = at(i, j), b = at(i + 1, j), d = at(i, j + 1), e = at(i + 1, j + 1);
                let n = V.norm(V.sub(V.mul(V.add(V.add(a, b), d), 1 / 3), center));
                this.tri(a, b, d, c, n);
                n = V.norm(V.sub(V.mul(V.add(V.add(d, b), e), 1 / 3), center));
                this.tri(d, b, e, c, n);
            } }
        cylinder(a, b, r, c) { const d = V.norm(V.sub(b, a)), x = V.norm(V.cross(d, Math.abs(d[1]) > .9 ? [1, 0, 0] : [0, 1, 0])), y = V.cross(d, x); for (let j = 0; j < 6; j++) {
            const n1 = V.add(V.mul(x, Math.cos(j * TAU / 6)), V.mul(y, Math.sin(j * TAU / 6))), n2 = V.add(V.mul(x, Math.cos((j + 1) * TAU / 6)), V.mul(y, Math.sin((j + 1) * TAU / 6))), p = V.add(a, V.mul(n1, r)), q = V.add(b, V.mul(n1, r)), s = V.add(a, V.mul(n2, r)), t = V.add(b, V.mul(n2, r));
            this.tri(p, q, s, c, n1);
            this.tri(s, q, t, c, n2);
        } }
        ring(p, r, color, axis = 'y') { let last = null; for (let i = 0; i <= 64; i++) {
            let a = i / 64 * TAU, q = axis === 'y' ? [p[0] + Math.cos(a) * r, p[1], p[2] + Math.sin(a) * r] : [p[0] + Math.cos(a) * r, p[1] + Math.sin(a) * r, p[2]];
            if (last)
                this.line(last, q, color);
            last = q;
        } }
    }
    function addFly(g,b,mode){
        if(!b.legs)return; // Never draw synthetic gait as measured physics.
        const brown=[.51,.34,.16,1], dark=[.19,.15,.12,1], blue=[.21,.92,.79,1];
        const basis=b.basis||[[1,0,0],[0,0,1],[0,-1,0]];
        // Geometry is a schematic skin on the measured segment transforms.
        const tr=p=>b.position.map((x,i)=>x+basis[i][0]*p[0]+basis[i][1]*p[1]+basis[i][2]*p[2]);
        const seg=b.segments||{};
        const at=center=>p=>center.map((x,i)=>x+basis[i][0]*p[0]+basis[i][1]*p[1]+basis[i][2]*p[2]);
        g.sphere([0,0,0],[.63,.36,.34],brown,at(seg.c_thorax||b.position));
        g.sphere([0,0,0],[.40,.31,.36],brown,at(seg.c_head||tr([.8,0,.1])));
        g.sphere([0,0,0],[.81,.33,.32],brown,at(seg.c_abdomen||tr([-.95,0,-.02])));
        const headAt=at(seg.c_head||tr([.8,0,.1]));
        for(const side of [-1,1]){
            g.sphere([.23,side*.26,.10],[.19,.22,.15],[.76,.20,.10,1],headAt);
            g.cylinder(headAt([.30,side*.12,.15]),headAt([.60,side*.23,.22]),.023,dark);
            const wing=[tr([.15,side*.25,.35]),tr([-.65,side*.8,.37]),tr([-1.70,side*.75,.28]),tr([-1.20,side*.28,.30])];
            g.tri(wing[0],wing[1],wing[2],[.64,.86,.86,.35]);g.tri(wing[0],wing[2],wing[3],[.64,.86,.86,.35]);
            for(let k=0;k<4;k++)g.line(wing[k],wing[(k+1)%4],[.55,.77,.75,.6]);
        }
        ['lf','lm','lh','rf','rm','rh'].forEach((leg,i)=>{
            const chain=b.legs[leg]||[];for(let k=1;k<chain.length;k++){if(V.len(V.sub(chain[k],chain[k-1]))>.00001)g.cylinder(chain[k-1],chain[k],k<3?.04:.025,k<3?brown:dark);}
            if(chain.length){const toe=chain[chain.length-1];g.sphere(toe,[.075,.055,.075],b.contactsBW[i]>.01?blue:[.6,.48,.3,1],x=>x,8,5);}
        });
    }
    const vert = `attribute vec3 p;attribute vec3 n;attribute vec4 c;uniform mat4 vp;varying vec3 vn;varying vec3 pos;varying vec4 col;void main(){gl_Position=vp*vec4(p,1.);vn=n;pos=p;col=c;}`;
    const frag = `precision mediump float;varying vec3 vn;varying vec3 pos;varying vec4 col;uniform vec3 eye;uniform float lit;void main(){vec3 cc=col.rgb;if(lit>.5){vec3 no=normalize(vn),l=normalize(vec3(-.4,.9,.6)),v=normalize(eye-pos);float s=pow(max(0.,dot(no,normalize(l+v))),30.);cc=cc*(.42+.65*max(0.,dot(no,l)))+vec3(.8,.9,.8)*s*.16;}float fog=smoothstep(28.,85.,distance(pos,eye))*.7;gl_FragColor=vec4(mix(cc,vec3(.026,.049,.061),fog),col.a);}`;
    class WorldView {
        constructor(canvas, overlay) {
            this.canvas = canvas;
            this.overlay = overlay;
            this.theta = .82;
            this.phi = .95;
            this.distance = 15;
            this.target = [-7, 3, -5];
            this.follow = true;
            this.firstPerson = false;
            this.showTrail = true;
            this.trail = new F.RingBuffer(1600);
            this.lastTick = -1;
            this.frame = null;
            this.dpr = Math.min(devicePixelRatio || 1, 1.75);
            this.engine = 'CANVAS 3D';
            this.disposed = false;
            this.renderStats = { frames: 0, geometryBuilds: 0, uploads: 0 };
            let gl = null;
            try {
                if (!(globalThis.FLY_FORCE_SOFTWARE || location.search.includes('software=1')))
                    gl = canvas.getContext('webgl', { alpha: false, antialias: true, preserveDrawingBuffer: true });
            }
            catch { }
            if (gl) {
                try {
                    this.initGL(gl);
                    this.engine = 'WEBGL';
                }
                catch (e) {
                    console.warn('WebGL fallback', e);
                    this.gl = null;
                    canvas.replaceWith(canvas.cloneNode());
                    this.canvas = document.querySelector('#world');
                    this.ctx = this.canvas.getContext('2d');
                }
            }
            else
                this.ctx = canvas.getContext('2d');
            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(this.canvas.parentElement);
            this.resize();
        }
        initGL(gl) { this.gl = gl; const shader = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS))
            throw Error(gl.getShaderInfoLog(x)); return x; }; this.program = gl.createProgram(); gl.attachShader(this.program, shader(gl.VERTEX_SHADER, vert)); gl.attachShader(this.program, shader(gl.FRAGMENT_SHADER, frag)); gl.linkProgram(this.program); if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
            throw Error('Shader link failure'); this.loc = { p: gl.getAttribLocation(this.program, 'p'), n: gl.getAttribLocation(this.program, 'n'), c: gl.getAttribLocation(this.program, 'c'), vp: gl.getUniformLocation(this.program, 'vp'), eye: gl.getUniformLocation(this.program, 'eye'), lit: gl.getUniformLocation(this.program, 'lit') }; this.buffer = gl.createBuffer(); this.lineBuffer = gl.createBuffer(); gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); this.canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); this.lost = true; window.dispatchEvent(new CustomEvent('fly-render-error', { detail: 'WebGL 컨텍스트가 중단되었습니다. 체크포인트를 저장하고 다시 여세요.' })); }); }
        resize() { this._drawnFrame = null; const r = this.canvas.getBoundingClientRect(); this.width = Math.max(1, r.width); this.height = Math.max(1, r.height); for (const c of [this.canvas, this.overlay]) {
            c.width = Math.round(this.width * this.dpr);
            c.height = Math.round(this.height * this.dpr);
        } if (this.gl)
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height); }
        setFrame(f) { if (f.tick < this.lastTick)
            this.trail.clear(); if (f.tick !== this.lastTick && (this.lastTick < 0 || f.tick % 12 < 4))
            this.trail.push([...f.body.position]); this.frame = f; this.lastTick = f.tick; }
        home() { this.follow = false; this.firstPerson = false; this.theta = .85; this.phi = .86; this.distance = 57; this.target = [0, 2.5, 0]; }
        track() { this.follow = true; this.firstPerson = false; this.distance = 19; this.phi = .91; }
        top() { this.firstPerson = false; this.follow = false; this.target = [0, 0, 0]; this.phi = .035; this.distance = 57; }
        camera() { if (this.follow && this.frame)
            this.target = V.lerp(this.target, this.frame.body.position, .12); if (this.firstPerson && this.frame) {
            const b = this.frame.body;
            this.eye = V.add(b.position, [Math.cos(b.yaw) * 1.6, .22, Math.sin(b.yaw) * 1.6]);
            this.lookAt = V.add(this.eye, [Math.cos(b.yaw) * 10, -.4, Math.sin(b.yaw) * 10]);
        }
        else {
            this.eye = V.add(this.target, [this.distance * Math.sin(this.phi) * Math.sin(this.theta), this.distance * Math.cos(this.phi), this.distance * Math.sin(this.phi) * Math.cos(this.theta)]);
            this.lookAt = this.target;
        } this.vp = M.mul(M.perspective(.78, this.width / this.height, .10, 180), M.look(this.eye, this.lookAt)); }
        project(p) { const r = M.transform(this.vp, p); return { x: (r[0] / r[3] * .5 + .5) * this.width, y: (.5 - r[1] / r[3] * .5) * this.height, z: r[3], visible: r[3] > .15 }; }
        pointOnPlane(x, y, h) { const f = V.norm(V.sub(this.lookAt, this.eye)), s = V.norm(V.cross(f, [0, 1, 0])), u = V.cross(s, f), t = Math.tan(.39), d = V.norm(V.add(f, V.add(V.mul(s, (x / this.width * 2 - 1) * t * this.width / this.height), V.mul(u, (1 - y / this.height * 2) * t)))); if (Math.abs(d[1]) < 1e-5)
            return null; const k = (h - this.eye[1]) / d[1]; return k > 0 ? V.add(this.eye, V.mul(d, k)) : null; }
        geometry() {
            const f = this.frame, w = f.world, b = f.body, g = new Geometry(), X = w.bounds[0], Z = w.bounds[2], floor = [.04, .082, .095, 1];
            g.tri([-X, 0, -Z], [X, 0, -Z], [-X, 0, Z], floor, [0, 1, 0]);
            g.tri([-X, 0, Z], [X, 0, -Z], [X, 0, Z], floor, [0, 1, 0]);
            for (let x = -X; x <= X; x += 2)
                g.line([x, .02, -Z], [x, .02, Z], [.16, .31, .34, x % 8 === 0 ? .42 : .16]);
            for (let z = -Z; z <= Z; z += 2)
                g.line([-X, .02, z], [X, .02, z], [.16, .31, .34, z % 8 === 0 ? .42 : .16]);
            const corners = [[-X, 0, -Z], [X, 0, -Z], [X, 0, Z], [-X, 0, Z]];
            for (let i = 0; i < 4; i++) {
                g.line(corners[i], corners[(i + 1) % 4], [.37, .60, .59, .45]);
                const j=(i+1)%4, upA=V.add(corners[i],[0,4,0]),upB=V.add(corners[j],[0,4,0]);
                g.line(corners[i],upA,[.24,.46,.47,.3]);g.line(upA,upB,[.24,.46,.47,.3]);
                g.tri(corners[i],corners[j],upA,[.16,.24,.29,.09]);g.tri(upA,corners[j],upB,[.16,.24,.29,.09]);
            }
            for (const o of w.obstacles)
                g.sphere(o.p, [o.r, o.r, o.r], [.16, .25, .29, 1], x => x, 12, 7);
            for (const s of w.sources) {
                const active = s.kind === 'hazard' || w.foodOn, c = s.kind === 'food' ? [.90, .66, .32] : [.85, .35, .34];
                g.line([s.p[0], .06, s.p[2]], s.p, [...c, .2]);
                g.ring(s.p, 1.1 + [Math.sin(f.simTime * .8) * .1], [...c, active ? .34 : .1]);
                g.ring([s.p[0], .06, s.p[2]], .65, [...c, .30]);
                g.sphere(s.p, [.23, .23, .23], [...c, active ? .9 : .25], x => x, 10, 6);
                for (let j = 0; j < 12; j++) {
                    let a = j * 2.4 + f.simTime * .3, q = V.add(s.p, [Math.cos(a) * (.5 + j * .035), Math.sin(a * .8) * .5, Math.sin(a) * (.5 + j * .035)]);
                    g.line(q, V.add(q, [0, .08, 0]), [...c, active ? .4 : .1]);
                }
            }
            // Same inner-wall stripe geometry used by the ray-receptor adapter.
            const lc=w.cueOn?[.12,.83,.72,.9]:[.16,.21,.26,.5], lx=X-.20;
            g.tri([lx,0,-2.8],[lx,6,-2.8],[lx,0,2.8],lc);
            g.tri([lx,0,2.8],[lx,6,-2.8],[lx,6,2.8],lc);
            if (this.showTrail) {
                const ps = this.trail.array();
                for (let i = 1; i < ps.length; i++)
                    g.line(ps[i - 1], ps[i], [.35, .91, .79, .08 + .45 * i / ps.length]);
            }
            // Projected soft footprint is illustrative, not an optical shadow solver.
            g.sphere([b.position[0], .035, b.position[2]], [1.55, .015, .72], [0, .02, .025, .32], x => x, 18, 3);
            if (!this.firstPerson)
                addFly(g, b, f.config.mode);
            return g;
        }
        drawGL(g) {
            const gl = this.gl;
            if (this.lost)
                return;
            gl.clearColor(.026, .049, .061, 1);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.useProgram(this.program);
            gl.uniformMatrix4fv(this.loc.vp, false, this.vp);
            gl.uniform3fv(this.loc.eye, this.eye);
            if (this._uploadedGeometry !== g) {
                const tris = [], lines = [];
                for (const t of g.triangles) for (const p of t.p) tris.push(...p, ...t.n, ...t.c);
                for (const l of g.lines) for (const p of l.p) lines.push(...p, 0, 1, 0, ...l.c);
                gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tris), gl.DYNAMIC_DRAW);
                gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.DYNAMIC_DRAW);
                this._triangleCount = tris.length / 10; this._lineCount = lines.length / 10;
                this._uploadedGeometry = g; this.renderStats.uploads += 2;
            }
            const bind = buffer => {
                gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
                for (const [n, size, off] of [['p', 3, 0], ['n', 3, 12], ['c', 4, 24]]) {
                    gl.enableVertexAttribArray(this.loc[n]);
                    gl.vertexAttribPointer(this.loc[n], size, gl.FLOAT, false, 40, off);
                }
            };
            bind(this.buffer);
            gl.uniform1f(this.loc.lit, 1);
            gl.drawArrays(gl.TRIANGLES, 0, this._triangleCount);
            bind(this.lineBuffer);
            gl.uniform1f(this.loc.lit, 0);
            gl.drawArrays(gl.LINES, 0, this._lineCount);
        }
        drawSoftware(g) { const ctx = this.ctx; ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); ctx.fillStyle = '#071019'; ctx.fillRect(0, 0, this.width, this.height); const jobs = []; const light = V.norm([-.4, .9, .6]); for (const t of g.triangles) {
            const p = t.p.map(v => this.project(v));
            if (p.some(v => !v.visible))
                continue;
            const center = V.mul(t.p.reduce((a, b) => V.add(a, b), [0, 0, 0]), 1 / 3);
            if (t.c[3] >= 1 && V.dot(t.n, V.sub(this.eye, center)) < 0)
                continue;
            jobs.push({ p, c: rgba(t.c, .42 + .65 * Math.max(0, V.dot(t.n, light))), z: p.reduce((s, v) => s + v.z, 0) / 3, tri: true });
        } for (const l of g.lines) {
            const p = l.p.map(v => this.project(v));
            if (p.some(v => !v.visible))
                continue;
            jobs.push({ p, c: rgba(l.c), z: (p[0].z + p[1].z) / 2, tri: false });
        } jobs.sort((a, b) => b.z - a.z); for (const j of jobs) {
            ctx.beginPath();
            ctx.moveTo(j.p[0].x, j.p[0].y);
            for (let i = 1; i < j.p.length; i++)
                ctx.lineTo(j.p[i].x, j.p[i].y);
            if (j.tri) {
                ctx.closePath();
                ctx.fillStyle = j.c;
                ctx.fill();
                ctx.strokeStyle = j.c;
                ctx.lineWidth = .35;
                ctx.stroke();
            }
            else {
                ctx.strokeStyle = j.c;
                ctx.lineWidth = .75;
                ctx.stroke();
            }
        } }
        hud() { const ctx = this.overlay.getContext('2d'); ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); ctx.clearRect(0, 0, this.width, this.height); ctx.font = '10px ui-monospace,monospace'; for (const s of this.frame.world.sources) {
            const p = this.project(s.p);
            if (!p.visible || p.x < 0 || p.x > this.width)
                continue;
            ctx.fillStyle = s.kind === 'food' ? '#ddb980' : '#d98c88';
            ctx.fillText(s.kind === 'food' ? 'ODOR +' : 'AVERSIVE', p.x + 10, p.y - 9);
        } const p = this.project(this.frame.body.position); if (!this.firstPerson && p.visible) {
            ctx.strokeStyle = '#79d6c944';
            ctx.beginPath();
            ctx.arc(p.x, p.y, 6, 0, TAU);
            ctx.stroke();
            ctx.fillStyle = '#a1bcb9';
            ctx.fillText('D. melanogaster · B', p.x + 15, p.y - 28);
        } ctx.fillStyle = '#618583'; ctx.fillText('x / z: mm   y: height   •   MODEL SPACE', 18, this.height - 18); }
        cameraKey() { return [this.theta, this.phi, this.distance, ...this.target, this.follow, this.firstPerson,
            this.width, this.height, this.showTrail].join(','); }
        needsDraw() {
            if (!this.frame || this.disposed || this.lost) return false;
            return this._drawnFrame !== this.frame || this._cameraKey !== this.cameraKey() ||
                (this.follow && !this.firstPerson && this.target.some((v, i) => Math.abs(v-this.frame.body.position[i]) > 1e-5));
        }
        draw() { if (!this.frame || this.disposed) return;
            this.camera();
            if (this._geometryFrame !== this.frame || this._geometryTrail !== this.showTrail || this._geometryFirstPerson !== this.firstPerson) {
                this._geometry = this.geometry(); this._geometryFrame = this.frame; this._geometryTrail = this.showTrail;
                this._geometryFirstPerson = this.firstPerson;
                this.renderStats.geometryBuilds++;
            }
            this.gl ? this.drawGL(this._geometry) : this.drawSoftware(this._geometry); this.hud();
            this._drawnFrame = this.frame; this._cameraKey = this.cameraKey(); this.renderStats.frames++;
            if (this.canvas.dataset) Object.assign(this.canvas.dataset, {renderFrames:String(this.renderStats.frames),
                geometryBuilds:String(this.renderStats.geometryBuilds),geometryUploads:String(this.renderStats.uploads)});
        }
        dispose() { this.disposed = true; this.resizeObserver.disconnect(); if (this.gl) {
            this.gl.deleteBuffer(this.buffer);
            this.gl.deleteBuffer(this.lineBuffer);
            this.gl.deleteProgram(this.program);
        } }
    }
    F.WorldView = WorldView;
    F.Geometry = Geometry;
})(globalThis.Fly);
