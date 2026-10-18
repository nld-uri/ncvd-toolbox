"""Browser-native versions of the NCVD deck animations (one <x-embed> page each)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'python'))
from embedkit import e16, e8mu, js, build
import ncvd as nc
from ncvd import systems as sy
OUT = os.path.join(HERE, 'embeds'); os.makedirs(OUT, exist_ok=True)


def complex_mode():
    M, Cm, K = sy.chain(6, 1, 4, 0.02, {0: 1.0, 1: 1.0})
    lam, E = sy.true_modes(M, Cm, K)
    i = 3
    psi = nc.normalize_shape(E[:, i]); l = lam[i]
    _, mp = nc.realify(psi)
    d = dict(pr=np.round(psi.real, 5).tolist(), pi=np.round(psi.imag, 5).tolist(), lr=round(l.real, 5), li=round(abs(l.imag), 5))
    code = f"""var D={js(d)},n=6,R=1.1*Math.max.apply(0,D.pr.map(function(r,j){{return Math.hypot(r,D.pi[j])}})),Tp=2.2*2*Math.PI/D.li;
var a1=new Ax(170,60,540,540,{{xl:[-R,R],yl:[-R,R],xlab:'Re',ylab:'Im',yoff:40,title:'$ψ_{{j}} e^{{λt}}$ in the complex plane'}});
var a2=new Ax(880,60,680,540,{{xl:[.5,6.5],yl:[-1.25*R,1.25*R],xt:[1,2,3,4,5,6].map(function(k){{return[k,''+k]}}),xlab:'station $j$',ylab:'$q_{{j}}(t)=$ Re $ψ_{{j}} e^{{λt}}$',yoff:44,title:'6-DOF chain, mode 4: $λ={l.real:.3f}+{abs(l.imag):.3f}$i,  MPC $={mp:.2f}$',ts:25}});
function Z(t){{var e=Math.exp(D.lr*t),c=Math.cos(D.li*t),s=Math.sin(D.li*t);return D.pr.map(function(r,j){{return[e*(r*c-D.pi[j]*s),e*(r*s+D.pi[j]*c)]}})}}
var X=[1,2,3,4,5,6];
run(14,function(p){{var t=p*Tp,z=Z(t);a1.frame();a2.frame();a1.hl(0);a1.vl(0,REF,[]);a2.hl(0);
a1.line([0].concat(z.map(function(q){{return q[0]}})),[0].concat(z.map(function(q){{return q[1]}})),{{c:C[0],w:3}});
z.forEach(function(q){{a1.line([q[0],q[0]],[q[1],0],{{c:C[1],w:1.6,d:[4,5]}});a1.dot(q[0],q[1],C[0],7);a1.dot(q[0],0,C[1],6)}});
for(var k=6;k>=1;k--){{var zz=Z(t-k*Tp/40);a2.line(X,zz.map(function(q){{return q[0]}}),{{c:REF,w:1.4,al:.55-k*.07}})}}
a2.line(X,z.map(function(q){{return q[0]}}),{{c:C[1],w:3}});z.forEach(function(q,j){{a2.dot(j+1,q[0],C[1],7)}});
tx('no common phase: the nodes travel',a2.x+16,a2.y+a2.h-18,{{s:24,c:REF}})}});"""
    build('complex_mode', code, OUT)


def joukowski():
    n = 4
    K = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1); K[-1, -1] = 1
    Cd = 0.02*np.eye(n) + 0.01*K
    lam = np.linalg.eigvals(sy.state_matrix(np.eye(n), Cd, K))
    pos = sorted([i for i in range(8) if lam[i].imag > 0], key=lambda i: abs(lam[i]))
    neg = [int(np.argmin(np.abs(lam - lam[i].conjugate()))) for i in pos]
    seq = [x for pair in zip(pos, neg) for x in pair]
    L = lam[seq]
    w = np.sort(np.abs(lam))[::2]
    t = np.arange(0, 120, 0.05)
    d, v, a = sy.simulate_linear(np.eye(n), Cd, K, np.random.default_rng(1).standard_normal(2*n), t)
    grid = np.unique(np.r_[np.geomspace(0.25, 2.6, 260), np.concatenate([x*np.geomspace(0.999, 1.001, 25) for x in w]), w])

    def offd(c):
        r = nc.ncvd(d, v/c, a/c**2)
        S = r.S2; D = np.diag(np.diag(S)); return np.linalg.norm(S - D)/np.linalg.norm(D)
    od = np.log10(np.maximum([offd(c) for c in grid], 1e-16))
    gap = np.log10(np.maximum([min(abs(0.5*(x/c + c/x) - 0.5*(y/c + c/y)) for i, x in enumerate(lam) for j, y in enumerate(lam) if i < j) for c in grid], 1e-16))
    D = dict(lr=np.round(L.real, 5).tolist(), li=np.round(L.imag, 5).tolist(), w=np.round(w, 5).tolist(),
             g=e16(np.log10(grid), 1e-4), od=e16(od, 1e-3), gp=e16(gap, 1e-3))
    code = f"""var D={js(D)},G=b64(D.g,1e-4).map(function(v){{return Math.pow(10,v)}}),OD=b64(D.od,1e-3).map(function(v){{return Math.pow(10,v)}}),GP=b64(D.gp,1e-3).map(function(v){{return Math.pow(10,v)}});
var a1=new Ax(120,60,400,560,{{xl:[-2.8,1.2],yl:[-2.8,2.8],xt:[[-2,'−2'],[0,'0']],yt:[[-2,'−2'],[0,'0'],[2,'2']],xlab:'Re $λ/c$',ylab:'Im $λ/c$',yoff:48,title:'poles, time unit $1/c$'}});
var a2=new Ax(650,120,440,440,{{xl:[-1.6,1.6],yl:[-1.6,1.6],xt:[[-1,'−1'],[0,'0'],[1,'1']],yt:[[-1,'−1'],[0,'0'],[1,'1']],xlab:'Re $μ$',ylab:'Im $μ$',yoff:48,title:'$μ=(λ/c+c/λ)/2$'}});
var a3=new Ax(1230,60,340,560,{{xl:[.25,2.6],yl:[1e-16,10],xlog:1,ylog:1,xt:[[.3,'0.3'],[1,'1'],[2,'2']],yt:[[1e-15,'$10^{{−15}}$'],[1e-10,'$10^{{−10}}$'],[1e-5,'$10^{{−5}}$'],[1,'1']],xlab:'$c$',yoff:80,title:'ties at $c=|λ_{{k}}|$'}});
var cir=lin(0,2*Math.PI,200),A=Math.log(.25),B=Math.log(2.6);
run(16,function(p){{var c=Math.exp(A+(B-A)*(p<.5?2*p:2-2*p));a1.frame();a2.frame();a3.frame();
a1.hl(0,REF,[],1);a1.vl(0,REF,[],1);a1.line(cir.map(Math.cos),cir.map(Math.sin),{{c:REF,d:[8,6],w:2}});
a2.hl(0,REF,[],1);a2.vl(0,REF,[],1);a2.line([-1,1],[0,0],{{c:REF,w:7,al:.5}});
D.w.forEach(function(x){{a3.vl(x,REF,[3,5],1.4)}});a3.line(G,GP,{{c:C[0],w:2.4}});a3.line(G,OD,{{c:C[1],w:2}});a3.vl(c,INK,[],2.2);
tx('smallest',1300,325,{{s:24,c:C[0]}});tx('gap of $μ$',1300,355,{{s:24,c:C[0]}});tx('off-diagonal',1300,505,{{s:24,c:C[1]}});tx('of $Σ_{{2}}$',1300,535,{{s:24,c:C[1]}});
for(var k=0;k<8;k++){{var zr=D.lr[k]/c,zi=D.li[k]/c,m=zr*zr+zi*zi,mr=.5*(zr+zr/m),mi=.5*(zi-zi/m),col=C[k>>1],op=k%2;a1.dot(zr,zi,col,8,op);a2.dot(mr,mi,col,8,op)}}
tx('$c=$'+c.toFixed(2),a1.x+16,a1.y+a1.h-18,{{s:26}})}});"""
    build('joukowski_fold', code, OUT)


def residual():
    code = r"""var P=[[1.3,1.3,0],[-1.3,1.3,0],[-1.3,-1.3,0],[1.3,-1.3,0]],Y=[.8,.6,.55];
function pr(v,az,el){var ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),x=-v[0]*sa+v[1]*ca,d=v[0]*ca+v[1]*sa;return[440+250*x,400-250*(v[2]*ce-d*se)]}
function arrow(a,b,c,w,dash){var d=Math.atan2(b[1]-a[1],b[0]-a[0]);g.strokeStyle=c;g.fillStyle=c;g.lineWidth=w;g.setLineDash(dash||[]);g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke();g.setLineDash([]);
if(!dash){g.beginPath();g.moveTo(b[0],b[1]);g.lineTo(b[0]-22*Math.cos(d-.35),b[1]-22*Math.sin(d-.35));g.lineTo(b[0]-22*Math.cos(d+.35),b[1]-22*Math.sin(d+.35));g.fill()}}
run(16,function(p){var az=(-60+90*(p<.5?2*p:2-2*p))*Math.PI/180,el=(22+8*Math.sin(2*Math.PI*p))*Math.PI/180,f=function(v){return pr(v,az,el)},o=f([0,0,0]);
g.beginPath();P.forEach(function(v,i){var q=f(v);i?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1])});g.closePath();g.fillStyle='rgba(31,90,133,.12)';g.fill();
arrow(o,f([1,0,0]),C[0],4);arrow(o,f([.3,.9,0]),C[0],4);arrow(o,f([Y[0],Y[1],0]),C[2],5);arrow(o,f(Y),C[1],5);arrow(f([Y[0],Y[1],0]),f(Y),C[4],4,[10,8]);
var q=f([1.08,-.05,0]);tx('$𝐱_{1}$-columns',q[0],q[1],{s:26,c:C[0]});q=f([Y[0]+.03,Y[1]+.03,Y[2]+.08]);tx('$ẋ$ (a rate column)',q[0],q[1],{s:26,c:C[1]});
q=f([Y[0]+.08,Y[1],0]);tx('$𝐔𝚲𝚽^{𝖧}$',q[0],q[1]+28,{s:28,c:C[2]});q=f([Y[0]+.06,Y[1]+.02,Y[2]/2]);tx('$𝐑$',q[0],q[1],{s:30,c:C[4]});
var x=960;tx('$𝐗_{1}=𝐔𝚽^{𝖧}$     exact',x,140,{s:34});tx('$𝐗_{2}=𝐔𝚲𝚽^{𝖧}+𝐑$',x,210,{s:34});
tx('$𝐑=(𝐈−𝐗_{1}𝐗_{1}^{†})𝐗_{2}$  ⊥  span $𝐗_{1}$',x,290,{s:32});
tx('linear, noise-free: $𝐑=𝟎$',x,400,{s:30,c:C[2]});tx('nonlinear window: $‖𝐑‖/‖𝐗_{2}‖$',x,460,{s:30,c:C[4]});
tx('is what no linear model of the window explains',x,510,{s:26,c:REF})});"""
    build('residual_projection', code, OUT)


def fir_band():
    import scipy.signal as ss
    sg = ss.savgol_coeffs(7, 3, deriv=1, use='dot').tolist()
    F = [[[-1, 1], [0, 1]], [[-.5, .5], [-1, 1]], [[1/12, -8/12, 8/12, -1/12], [-2, -1, 1, 2]], [[round(c, 8) for c in sg], list(range(-3, 4))]]
    code = f"""var F={js(F)},NM=['forward','central','five-point','Savitzky–Golay (7,3)'],PI=Math.PI;
function sym(f,w){{var r=0,i=0;f[0].forEach(function(c,j){{r+=c*Math.cos(f[1][j]*w);i+=c*Math.sin(f[1][j]*w)}});return[r,i]}}
var ws=lin(.001,PI,400),cur=F.map(function(f,k){{return ws.map(function(w){{var s=sym(f,w);return k?s[1]:Math.hypot(s[0],s[1])}})}}),
band=F.map(function(f,k){{if(!k)return PI;var y=cur[k];for(var i=1;i<y.length;i++)if(y[i]<=y[i-1])return ws[i];return PI}});
var a1=new Ax(110,130,440,440,{{xl:[-1.3,1.3],yl:[-1.3,1.3],xt:[[-1,'−1'],[0,'0'],[1,'1']],yt:[[-1,'−1'],[0,'0'],[1,'1']],xlab:'Re $θ$',ylab:'Im $θ$',yoff:48,title:'$θ=e^{{iωΔt}}$'}});
var a2=new Ax(720,60,840,540,{{xl:[0,PI],yl:[0,2.2],xt:[[0,'0'],[PI/4,'$π/4$'],[PI/2,'$π/2$'],[3*PI/4,'$3π/4$'],[PI,'$π$']],yt:[[0,'0'],[.5,'0.5'],[1,'1'],[1.5,'1.5'],[2,'2']],xlab:'$ωΔt$',ylab:'Im $d(θ)Δt$  (forward: $|d|Δt$)',yoff:56,title:'the correction is unique while the curve rises (dotted: end of band)',ts:25}});
var cir=lin(0,2*PI,200);
run(12,function(p){{var w=.02+(PI-.04)*(p<.5?2*p:2-2*p);a1.frame();a2.frame();a1.hl(0,REF,[],1);a1.vl(0,REF,[],1);a1.line(cir.map(Math.cos),cir.map(Math.sin),{{c:REF,w:1.6}});
var ar=lin(0,w,40);a1.line(ar.map(function(t){{return .35*Math.cos(t)}}),ar.map(function(t){{return .35*Math.sin(t)}}),{{c:INK,w:3}});a1.dot(Math.cos(w),Math.sin(w),INK,9);
a2.line(ws,ws,{{c:REF,d:[8,6],w:2}});F.forEach(function(f,k){{a2.line(ws,cur[k],{{c:C[k],w:3}});if(band[k]<PI-1e-3)a2.vl(band[k],C[k],[3,5],2);var s=sym(f,w);a2.dot(w,k?s[1]:Math.hypot(s[0],s[1]),C[k],8)}});
leg(a2.x+24,a2.y+40,[['exact, $ωΔt$',REF,[8,6],2]].concat(NM.map(function(n,k){{return[n,C[k]]}})));
tx('$ωΔt=$'+w.toFixed(2),a1.x+14,a1.y+a1.h-16,{{s:26}})}});"""
    build('fir_band', code, OUT)


def backbone():
    M, _, K, fnl = sy.duffing_chain(3, k=1.0, c=0.0, k3={0: 1.0}); Cd = 0.003*M
    lam0, E0 = sy.true_modes(M, Cd, K)
    sh = [nc.realify(E0[:, i])[0] for i in range(3)]; sh = [x/np.abs(x).max() for x in sh]
    x0 = np.r_[1.5*sh[0] + 0.6*sh[1] + 0.5*sh[2], np.zeros(3)]
    fs = 20.0; t = np.arange(0, 1500, 1/fs)
    d, v, a = sy.simulate_nonlinear(M, Cd, K, fnl, x0, t)
    L, S = 800, 200
    tr = nc.track(d, v, a, fs=fs, length=L, step=S, n_track=3, M=M)
    E = tr.energy; rat = tr.wd/np.abs(lam0.imag)[None, :]
    dec = 7                                           # 20/6 = 3.33 Hz record for the plots
    q = d[::dec, 0]; amp = np.abs(q).max()
    D = dict(fs=fs/dec, n=len(q), q=e8mu(q, amp), A=round(float(amp), 5), ed=(tr.edges[:, 0]/dec).astype(int).tolist(), L=L//dec,
             E=e16(np.log10(E), 1e-3), r=e16(rat - 1, 2e-5), rs=e16(np.log10(tr.res), 1e-3),
             El=[float(np.nanmin(E))*.7, float(np.nanmax(E))*1.5], rm=float(np.nanmax(rat))*1.02,
             Rl=[float(np.nanmin(tr.res))*.5, float(np.nanmax(tr.res))*2], T=float(t[-1]))
    print('windows', len(tr.t))
    code = f"""var D={js(D)},q=mu(b64(D.q,1,0,8),D.A),n=D.n,dt=1/D.fs,tt=lin(0,(n-1)*dt,n),nw=D.ed.length,
E=b64(D.E,1e-3).map(function(v){{return Math.pow(10,v)}}),R=b64(D.r,2e-5,1),RS=b64(D.rs,1e-3).map(function(v){{return Math.pow(10,v)}}),
TW=D.ed.map(function(e){{return(e+D.L/2)*dt}}),v=q.map(function(x,i){{return i&&i<n-1?(q[i+1]-q[i-1])/(2*dt):0}});
function col(A,j){{var r=[];for(var i=0;i<nw;i++)r.push(A[3*i+j]);return r}}
var a1=new Ax(110,40,1450,220,{{xl:[0,D.T],yl:[-1.1*D.A,1.1*D.A],ylab:'$q_{{1}}$',yoff:30,title:'one free decay; the shaded window is decomposed',ts:25}}),
a2=new Ax(110,390,300,220,{{xl:[0,1],yl:[0,1],xlab:'$q_{{1}}$',ylab:'$q̇_{{1}}$',yoff:24,title:'window orbit',ts:25}}),
a3=new Ax(640,390,400,220,{{xl:D.El,yl:[.98,D.rm],xlog:1,yt:[[1,'1'],[1.05,'1.05'],[1.1,'1.1'],[1.15,'1.15'],[1.2,'1.2']].filter(function(t){{return t[0]<D.rm}}),xlab:'modal energy',ylab:'$ω_{{d}}/ω_{{d,lin}}$',yoff:78,title:'backbone curves',ts:25}}),
a4=new Ax(1250,390,310,220,{{xl:[0,D.T],yl:D.Rl,ylog:1,xt:[[0,'0'],[500,'500'],[1000,'1000']],xlab:'$t$ (s)',ylab:'$‖𝐑‖/‖𝐗_{{2}}‖$',yoff:44,title:'window residual',ts:25}});
var m=[0,1,2].map(function(j){{return[col(E,j),col(R,j)]}});
run(18,function(p){{var w=Math.min(nw-1,Math.floor(p*1.1*nw)),i0=D.ed[w],i1=i0+D.L;
a1.frame();a1.clip(function(){{g.fillStyle='rgba(180,98,42,.25)';g.fillRect(a1.X(tt[i0]),a1.y,a1.X(tt[i1])-a1.X(tt[i0]),a1.h)}});a1.line(tt,q,{{c:C[0],w:1}});tx('$t$ (s)',a1.x+a1.w-10,a1.y+30,{{a:'r',s:24}});
var qm=0,vm=0;for(var i=i0;i<i1;i++){{qm=Math.max(qm,Math.abs(q[i]));vm=Math.max(vm,Math.abs(v[i]))}}a2.xl=[-1.1*qm,1.1*qm];a2.yl=[-1.1*vm,1.1*vm];a2.frame();a2.line(q,v,{{c:C[0],w:1.6,sm:1}},i0,i1);
a3.frame();a3.hl(1,REF,[8,6],2);m.forEach(function(s,j){{a3.line(s[0],s[1],{{c:C[j],w:2.4}},0,w+1);for(var k=0;k<=w;k+=1)a3.dot(s[0][k],s[1][k],C[j],3.5)}});
leg(a3.x+16,a3.y+34,[['mode 1',C[0]],['mode 2',C[1]],['mode 3',C[2]],['linear',REF,[8,6],2]],22);
a4.frame();a4.line(TW,RS,{{c:C[4],w:2.4}},0,w+1);for(k=0;k<=w;k++)a4.dot(TW[k],RS[k],C[4],3.5)}});"""
    build('backbone_tracking', code, OUT)


if __name__ == '__main__':
    names = sys.argv[1:] or ['complex_mode', 'joukowski', 'residual', 'fir_band', 'backbone']
    for nm in names:
        globals()[nm]()
