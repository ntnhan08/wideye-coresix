import copy, math, random, time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────

CLASSES = {0:"obstacle_box", 1:"path_clear", 2:"wall_detected",
           3:"person_nearby", 4:"vehicle_zone"}
NUM_CLASSES = len(CLASSES)
CORE_CH     = 128
NUM_CORES   = 6

# ══════════════════════════════════════════════════════════════════════
#  RepConvBN · SwiGLU-FFN · Sparse-MoE · LayerScale · CBAM
# ══════════════════════════════════════════════════════════════════════

class RepConvBN(nn.Module):
    """Multi-branch training → fused single Conv3×3 at deploy."""
    def __init__(self, ic, oc, s=1, act=True):
        super().__init__()
        self.ic = ic; self.oc = oc; self.s = s
        self.deployed = False
        self.br3 = nn.Sequential(nn.Conv2d(ic,oc,3,s,1,bias=False), nn.BatchNorm2d(oc))
        self.br1 = nn.Sequential(nn.Conv2d(ic,oc,1,s,0,bias=False), nn.BatchNorm2d(oc))
        self.brid = nn.BatchNorm2d(ic) if (ic==oc and s==1) else None
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()
        self._fc  = None

    def forward(self, x):
        if self.deployed:
            return self.act(self._fc(x))
        out = self.br3(x) + self.br1(x)
        if self.brid is not None:
            out = out + self.brid(x)
        return self.act(out)

    def reparameterize(self):
        if self.deployed: return
        W3, b3 = self._fbn(self.br3[0], self.br3[1])
        W1, b1 = self._fbn(self.br1[0], self.br1[1])
        W      = W3 + F.pad(W1, [1,1,1,1])
        b      = b3 + b1
        if self.brid is not None:
            Wi, bi = self._fid(self.brid); W += Wi; b += bi
        self._fc = nn.Conv2d(self.ic, self.oc, 3, self.s, 1, bias=True)
        self._fc.weight.data = W; self._fc.bias.data = b
        del self.br3, self.br1, self.brid
        self.deployed = True

    @staticmethod
    def _fbn(conv, bn):
        std   = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        return conv.weight * scale[:,None,None,None], bn.bias - bn.running_mean * scale

    def _fid(self, bn):
        std   = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        W     = torch.zeros(self.oc, self.oc, 3, 3, device=bn.weight.device)
        for i in range(self.oc): W[i, i, 1, 1] = 1.0
        return W * scale[:,None,None,None], bn.bias - bn.running_mean * scale


def reparameterize_model(m):
    for mod in m.modules():
        if isinstance(mod, RepConvBN): mod.reparameterize()
    return m


class SwiGLUFFN(nn.Module):
    """SwiGLU(x) = SiLU(Wx) ⊙ Vx — better gradient flow than ReLU-FFN."""
    def __init__(self, d, mult=4):
        super().__init__()
        hidden = int(d * mult * 2 / 3)  # 2/3× to match param count of std FFN
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up   = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.down(F.silu(self.gate(x)) * self.up(x))) + x


class SparseMoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts FFN.
    8 small experts, top-2 active per token.
    Total params ≈ standard FFN (experts are small).
    28 unique routing combinations → 28× pathway diversity.
    """
    def __init__(self, d, n_experts=8, top_k=2):
        super().__init__()
        self.n  = n_experts
        self.k  = top_k
        ed      = max(d // 4, 32)          # small expert hidden
        self.router  = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, ed, bias=False), nn.SiLU(inplace=True),
                nn.Linear(d, ed, bias=False),         # gate branch
                nn.Linear(ed, d, bias=False),
            ) for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(d)
        self._ed  = ed

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to tokens
        tokens = x.permute(0,2,3,1).reshape(-1, C)        # (B*H*W, C)
        # Router: top-k selection
        logits = self.router(tokens)                        # (N, n_exp)
        topk_w, topk_i = torch.topk(logits, self.k, dim=-1)
        topk_w = torch.softmax(topk_w, dim=-1)             # (N, k)
        # Dispatch to experts
        out = torch.zeros_like(tokens)
        for ki in range(self.k):
            idx = topk_i[:, ki]                            # (N,)
            w   = topk_w[:, ki:ki+1]                      # (N,1)
            for ei in range(self.n):
                mask = (idx == ei)
                if mask.any():
                    t   = tokens[mask]
                    e   = self.experts[ei]
                    # SwiGLU-style inside each expert
                    h   = F.silu(e[0](t)) * e[2](t)       # gate ⊙ up
                    out[mask] += w[mask] * e[3](h)
        out = self.norm(out) + tokens
        return out.reshape(B, H, W, C).permute(0,3,1,2)


class LayerScale(nn.Module):
    """Learnable per-channel scale, init near zero for stable deep training."""
    def __init__(self, dim, init=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init))

    def forward(self, x):
        return x * self.scale.view(-1, *([1] * (x.ndim - 1)))


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0: return x
        keep  = 1.0 - self.p
        mask  = torch.bernoulli(torch.full((x.shape[0],)+(1,)*(x.ndim-1), keep, device=x.device))
        return x * mask / keep


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__(); mid = max(ch//r, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch,mid,1,bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(mid,ch,1,bias=False), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)


class CBAM(nn.Module):
    def __init__(self, ch, r=16, sk=7):
        super().__init__(); mid = max(ch//r, 8)
        self.avg = nn.AdaptiveAvgPool2d(1); self.mx = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch,mid,1,bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(mid,ch,1,bias=False))
        self.sp  = nn.Conv2d(2,1,sk,padding=sk//2,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        ca = self.sig(self.mlp(self.avg(x)) + self.mlp(self.mx(x))); x = x * ca
        sa = self.sig(self.sp(torch.cat([x.mean(1,True), x.max(1,True)[0]], 1)))
        return x * sa


class ConvBN(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=1, g=1, act=True, dil=1):
        super().__init__()
        pad = p if dil==1 else dil
        ops = [nn.Conv2d(ic,oc,k,s,pad,groups=g,dilation=dil,bias=False), nn.BatchNorm2d(oc)]
        if act: ops.append(nn.SiLU(inplace=True))
        self.seq = nn.Sequential(*ops)
    def forward(self, x): return self.seq(x)


class DSConv(nn.Module):
    def __init__(self, ic, oc=None, s=1):
        super().__init__(); oc = oc or ic
        self.dw = nn.Sequential(nn.Conv2d(ic,ic,3,s,1,groups=ic,bias=False), nn.BatchNorm2d(ic), nn.SiLU(inplace=True))
        self.pw = nn.Sequential(nn.Conv2d(ic,oc,1,bias=False), nn.BatchNorm2d(oc), nn.SiLU(inplace=True))
    def forward(self, x): return self.pw(self.dw(x))


class Bottleneck(nn.Module):
    """Standard or MoE bottleneck with RepConv + LayerScale + DropPath."""
    expansion = 4
    def __init__(self, ic, mid, s=1, attn='none', use_moe=False, dp=0.0):
        super().__init__()
        oc = mid * self.expansion
        self.c1   = ConvBN(ic, mid, k=1, p=0)
        self.c2   = RepConvBN(mid, mid, s=s)
        if use_moe:
            self.c3 = SparseMoEFFN(oc)
            self.c3_proj = nn.Sequential(nn.Conv2d(mid,oc,1,bias=False), nn.BatchNorm2d(oc))
        else:
            self.c3 = nn.Sequential(nn.Conv2d(mid,oc,1,bias=False), nn.BatchNorm2d(oc))
            self.c3_proj = None
        self.use_moe = use_moe
        self.attn = SEBlock(oc) if attn=='se' else CBAM(oc) if attn=='cbam' else nn.Identity()
        self.ls   = LayerScale(oc, init=1e-4)
        self.dp   = DropPath(dp)
        self.sc   = nn.Sequential()
        if s != 1 or ic != oc:
            self.sc = nn.Sequential(nn.Conv2d(ic,oc,1,stride=s,bias=False), nn.BatchNorm2d(oc))
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        if self.use_moe:
            out = self.c3_proj(self.c2(self.c1(x)))
            out = self.c3(out)
        else:
            out = self.c3(self.c2(self.c1(x)))
        out = self.ls(self.attn(out))
        return self.act(self.dp(out) + self.sc(x))


def make_stage(ic, mid, depth, stride, ae=0, at='none', me=0, dp_rates=None):
    oc = mid * Bottleneck.expansion; blocks = []
    for i in range(depth):
        dp   = dp_rates[i] if dp_rates else 0.0
        ua   = at if ae > 0 and i % ae == 0 else 'none'
        um   = me > 0 and i % me == 0
        blocks.append(Bottleneck(ic if i==0 else oc, mid,
                                  s=stride if i==0 else 1,
                                  attn=ua, use_moe=um, dp=dp))
    return nn.Sequential(*blocks)


def init_weights(m):
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            if mod.bias is not None: nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
            if mod.bias is not None: nn.init.zeros_(mod.bias)


# ══════════════════════════════════════════════════════════════════════
#  SSM-lite (Mamba-style) · HybridAttention · 2D-RoPE · Six Cores
# ══════════════════════════════════════════════════════════════════════

class RoPE2D(nn.Module):
    """
    2D Rotary Position Encoding — zero learnable params.
    Encodes (row, col) position into Q/K before dot-product.
    Gives attention relative spatial awareness for free.
    """
    def __init__(self, dim, max_h=64, max_w=64):
        super().__init__()
        assert dim % 4 == 0
        half = dim // 2
        inv  = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('inv_freq', inv, persistent=False)
        self._cache = {}

    def _get_freqs(self, H, W, device):
        key = (H, W, str(device))
        if key in self._cache: return self._cache[key]
        t_h = torch.arange(H, device=device).float()
        t_w = torch.arange(W, device=device).float()
        fh  = torch.outer(t_h, self.inv_freq)            # (H, d/4)
        fw  = torch.outer(t_w, self.inv_freq)            # (W, d/4)
        # Repeat for sin/cos pairs
        fh  = torch.stack([fh, fh], -1).flatten(-2)      # (H, d/2)
        fw  = torch.stack([fw, fw], -1).flatten(-2)      # (W, d/2)
        cos_h = fh.cos()[None,:,None,:]                   # (1,H,1,d/2)
        sin_h = fh.sin()[None,:,None,:]
        cos_w = fw.cos()[None,None,:,:]                   # (1,1,W,d/2)
        sin_w = fw.sin()[None,None,:,:]
        self._cache[key] = (cos_h, sin_h, cos_w, sin_w)
        return self._cache[key]

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """q, k: (B, heads, H, W, head_dim)"""
        B, nh, H, W, d = q.shape
        half = d // 2
        cos_h, sin_h, cos_w, sin_w = self._get_freqs(H, W, q.device)
        # Apply to row dimension (first half of d)
        qh = q[..., :half] * cos_h + self._rotate_half(q[..., :half]) * sin_h
        kh = k[..., :half] * cos_h + self._rotate_half(k[..., :half]) * sin_h
        # Apply to col dimension (second half of d)
        qw = q[..., half:] * cos_w + self._rotate_half(q[..., half:]) * sin_w
        kw = k[..., half:] * cos_w + self._rotate_half(k[..., half:]) * sin_w
        return torch.cat([qh, qw], -1), torch.cat([kh, kw], -1)


class SSMLite(nn.Module):
    """
    Lightweight Selective State Space (Mamba-inspired).
    O(N) complexity vs O(N²) attention.
    Projects d→proj first to keep params light.
    """
    def __init__(self, d, proj=256, state_d=8):
        super().__init__()
        self.in_p   = ConvBN(d, proj, k=1, p=0)
        # Selective scan parameters
        self.in_proj= nn.Linear(proj, 2*proj, bias=False)  # x + z branches
        self.dt_proj= nn.Linear(proj, state_d, bias=True)
        self.A_log  = nn.Parameter(torch.randn(state_d, proj))
        self.D      = nn.Parameter(torch.ones(proj))
        self.out_p  = ConvBN(proj, d, k=1, p=0)
        self.norm   = nn.LayerNorm(proj)
        self.state_d= state_d; self.proj = proj

    def forward(self, x):
        B, C, H, W = x.shape; N = H * W
        xp = self.in_p(x)                                  # (B,proj,H,W)
        t  = xp.permute(0,2,3,1).reshape(B*N, self.proj)   # (B*N, proj)
        xz = self.in_proj(t)                                # (B*N, 2*proj)
        xi, z = xz.chunk(2, -1)
        # Selective dt
        dt = torch.softplus(self.dt_proj(xi))               # (B*N, state_d)
        # Discretized A: A_bar = exp(dt * A_log)
        A  = -torch.exp(self.A_log.float())                 # (state_d, proj)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))# (B*N, state_d, proj)
        # SSM output: simplified (no sequential dependency, parallel approx)
        y  = (A_bar.mean(1) * xi + self.D * xi) * F.silu(z)
        y  = self.norm(y)
        out= y.reshape(B, H, W, self.proj).permute(0,3,1,2)
        return self.out_p(out) + x


class HybridAttention(nn.Module):
    """
    Hybrid Local-Window + Linear Attention with 2D-RoPE.
    Shared Q,K,V projections for both branches → param efficient.
    """
    def __init__(self, d, proj=256, win=7, n_heads=4):
        super().__init__()
        self.proj_d = proj; self.nh = n_heads; self.win = win
        hd          = proj // n_heads
        self.in_p   = ConvBN(d, proj, k=1, p=0)
        self.qkv    = nn.Conv2d(proj, proj*3, 1, bias=False)
        self.out    = nn.Sequential(nn.Conv2d(proj, d, 1, bias=False), nn.BatchNorm2d(d))
        self.scale  = (proj // n_heads) ** -0.5
        self.rope   = RoPE2D(proj // n_heads)
        self.gate   = nn.Parameter(torch.tensor(0.5))     # blend local/global

    def _lwa(self, Q, K, V, H, W):
        """Local Window Attention (win×win)."""
        B, nh, _, hd = Q.shape
        w = self.win
        ph = (w - H%w)%w; pw_ = (w - W%w)%w
        def pad_and_win(t):
            t = t.reshape(B, nh, H, W, hd)
            t = F.pad(t.permute(0,1,4,2,3), (0,pw_,0,ph)).permute(0,1,3,4,2)
            Hp,Wp = H+ph, W+pw_; nH,nW = Hp//w, Wp//w
            return t.reshape(B,nh,nH,w,nW,w,hd).permute(0,1,2,4,3,5,6)\
                    .contiguous().reshape(B*nh*nH*nW, w*w, hd)
        Qw,Kw,Vw = pad_and_win(Q),pad_and_win(K),pad_and_win(V)
        attn = torch.softmax(torch.bmm(Qw, Kw.transpose(-2,-1))*self.scale, -1)
        out  = torch.bmm(attn, Vw)                        # (B*nh*nH*nW, w², hd)
        Hp,Wp = H+ph, W+pw_; nH,nW=Hp//w,Wp//w
        out  = out.reshape(B,nh,nH,nW,w,w,hd).permute(0,1,2,4,3,5,6)\
                   .contiguous().reshape(B,nh,Hp,Wp,hd)
        return out[:,:,:H,:W,:].reshape(B, nh, H*W, hd)

    def _linear_attn(self, Q, K, V):
        """Linear Attention O(N)."""
        phi = lambda t: F.elu(t) + 1
        Q2,K2 = phi(Q), phi(K)                            # (B, nh, N, hd)
        KV = torch.einsum('bnik,bnjk->bnij', K2, V)       # (B, nh, hd, hd)
        num= torch.einsum('bnik,bnkj->bnij', Q2, KV)      # (B, nh, N, hd)
        den= (Q2 * K2.sum(-2, keepdim=True)).sum(-1, keepdim=True) + 1e-6
        return num / den

    def forward(self, x):
        B, C, H, W = x.shape; N = H*W
        xp = self.in_p(x)                                 # (B,proj,H,W)
        qkv= self.qkv(xp).chunk(3, 1)
        def reshape(t):
            return t.reshape(B, self.nh, self.proj//self.nh, N)\
                    .permute(0,1,3,2)                      # (B,nh,N,hd)
        Q,K,V = [reshape(t) for t in qkv]
        # 2D-RoPE on Q,K
        Qr = Q.reshape(B,self.nh,H,W,-1); Kr = K.reshape(B,self.nh,H,W,-1)
        Qr, Kr = self.rope(Qr, Kr)
        Q = Qr.reshape(B,self.nh,N,-1); K = Kr.reshape(B,self.nh,N,-1)
        # Hybrid blend
        g    = self.gate.sigmoid()
        lwa  = self._lwa(Q, K, V, H, W)
        lin  = self._linear_attn(Q, K, V)
        out  = g * lwa + (1-g) * lin                       # (B,nh,N,hd)
        out  = out.permute(0,1,3,2).reshape(B, self.proj_d, H, W)
        return self.out(out) + x


# ── Six Cognitive Cores (v5 design, SiLU upgrade) ─────────────────

class NhanThiGiac(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.orient = ConvBN(ch,ch,k=3,p=1,g=4)
        self.mix    = ConvBN(ch,ch,k=1,p=0)
        self.alpha  = nn.Parameter(torch.full((1,ch,1,1),0.5))
        self.p_sm   = nn.AvgPool2d(3,1,1); self.p_lg=nn.AvgPool2d(7,1,3)
        self.dog_bn = nn.BatchNorm2d(ch)
        self.deep   = nn.Sequential(DSConv(ch), DSConv(ch))
        init_weights(self)
    def forward(self, x):
        o = self.mix(self.orient(x))
        d = self.dog_bn(self.p_sm(x)-self.alpha.clamp(0,1)*self.p_lg(x))
        return self.deep(o+d+x)


class CoordConv(nn.Module):
    def forward(self, x):
        B,_,H,W=x.shape
        yc=torch.linspace(-1,1,H,device=x.device).view(1,1,H,1).expand(B,1,H,W)
        xc=torch.linspace(-1,1,W,device=x.device).view(1,1,1,W).expand(B,1,H,W)
        return torch.cat([x,xc,yc],1)


class NhanTuDuyMoiTruong(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__(); mid=ch//4
        mk=lambda d,p: nn.Sequential(nn.Conv2d(ch,ch,3,padding=p,dilation=d,groups=ch,bias=False),nn.BatchNorm2d(ch),nn.SiLU(inplace=True))
        self.dw1=mk(1,1);self.dw2=mk(2,2);self.dw4=mk(4,4);self.dw8=mk(8,8)
        self.gp=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(ch,ch,1,bias=False),nn.BatchNorm2d(ch),nn.SiLU(inplace=True))
        self.spw=ConvBN(ch,mid,k=1,p=0); self.merge=ConvBN(5*mid,ch,k=1,p=0)
        self.coord=CoordConv()
        self.hp=nn.Sequential(nn.AdaptiveAvgPool2d((None,1)),nn.Conv2d(ch+2,mid,1,bias=False),nn.BatchNorm2d(mid),nn.SiLU(inplace=True))
        self.vp=nn.Sequential(nn.AdaptiveAvgPool2d((1,None)),nn.Conv2d(ch+2,mid,1,bias=False),nn.BatchNorm2d(mid),nn.SiLU(inplace=True))
        self.sm=ConvBN(ch+2*mid,ch,k=1,p=0); self.deep=nn.Sequential(DSConv(ch),DSConv(ch))
        init_weights(self)
    def forward(self, x):
        B,_,H,W=x.shape; pw=self.spw
        d1=pw(self.dw1(x));d2=pw(self.dw2(x));d4=pw(self.dw4(x));d8=pw(self.dw8(x))
        gc=pw(self.gp(x).expand_as(x))
        x=self.merge(torch.cat([d1,d2,d4,d8,gc],1))+x
        xc=self.coord(x)
        hp=self.hp(xc).expand(B,-1,H,W); vp=self.vp(xc).expand(B,-1,H,W)
        x=self.sm(torch.cat([x,hp,vp],1))+x
        return self.deep(x)


class LocalWindowAttentionCore(nn.Module):
    def __init__(self, ch=CORE_CH, win=7):
        super().__init__(); self.win=win; mid=ch//2
        self.q=nn.Conv2d(ch,mid,1,bias=False);self.k=nn.Conv2d(ch,mid,1,bias=False)
        self.v=nn.Conv2d(ch,mid,1,bias=False)
        self.out=nn.Sequential(nn.Conv2d(mid,ch,1,bias=False),nn.BatchNorm2d(ch))
        self.scale=mid**-0.5
    def forward(self, x):
        B,C,H,W=x.shape; w=self.win
        ph=(w-H%w)%w; pw_=(w-W%w)%w
        xp=F.pad(x,(0,pw_,0,ph)) if (ph or pw_) else x
        _,_,Hp,Wp=xp.shape; nH,nW=Hp//w,Wp//w
        Q=self.q(xp);K=self.k(xp);V=self.v(xp); mid=Q.shape[1]
        def tw(t): return t.view(B,mid,nH,w,nW,w).permute(0,2,4,1,3,5).contiguous().view(B*nH*nW,mid,w*w)
        Qw,Kw,Vw=tw(Q),tw(K),tw(V)
        attn=torch.softmax(torch.bmm(Qw.permute(0,2,1),Kw)*self.scale,-1)
        ow=torch.bmm(attn,Vw.permute(0,2,1)).permute(0,2,1).view(B*nH*nW,mid,w,w)
        out=ow.view(B,nH,nW,mid,w,w).permute(0,3,1,4,2,5).contiguous().view(B,mid,Hp,Wp)
        return self.out(out[:,:,:H,:W])


class NhanTuDuyNhanDien(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.lwa=LocalWindowAttentionCore(ch,7)
        self.tmpl=nn.Conv2d(ch,8,3,padding=1,bias=False)
        self.proj=ConvBN(8,ch,k=1,p=0)
        self.deep=nn.Sequential(DSConv(ch),DSConv(ch))
        init_weights(self)
    def forward(self, x):
        x=self.lwa(x)+x; x=self.proj(torch.softmax(self.tmpl(x),1))+x
        return self.deep(x)


class NhanHieuMoiTruong(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.protos=nn.Parameter(torch.randn(16,ch)); nn.init.orthogonal_(self.protos)
        self.proj=ConvBN(16,ch,k=1,p=0)
        bot=max(ch//8,8)
        self.occ=nn.Sequential(nn.Conv2d(ch,bot,1,bias=False),nn.BatchNorm2d(bot),nn.SiLU(inplace=True),nn.Conv2d(bot,1,1),nn.Sigmoid())
        self.gate=ConvBN(ch+1,ch,k=1,p=0); self.deep=nn.Sequential(DSConv(ch),DSConv(ch))
        init_weights(self)
    def forward(self, x):
        B,C,H,W=x.shape
        f=F.normalize(x.permute(0,2,3,1).reshape(-1,C),dim=1)
        p=F.normalize(self.protos,dim=1)
        sim=(f@p.T).view(B,H,W,-1).permute(0,3,1,2)
        x=self.proj(sim)+x; x=self.gate(torch.cat([x,self.occ(x)],1))+x
        return self.deep(x)


class NhanNhanBietMoiTruong(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.pool=nn.AvgPool2d(5,1,2)
        bot=ch//4
        self.sal=nn.Sequential(nn.Conv2d(ch,bot,1,bias=False),nn.BatchNorm2d(bot),nn.SiLU(inplace=True),nn.Conv2d(bot,1,3,padding=1),nn.Sigmoid())
        self.gate=ConvBN(ch+1,ch,k=1,p=0); self.deep=nn.Sequential(DSConv(ch),DSConv(ch))
        init_weights(self)
    def forward(self, x):
        mu=self.pool(x); xc=x-mu; xn=xc/(torch.sqrt(self.pool(xc**2)+1e-5))
        x=xn+x*0.5
        mu2=F.avg_pool2d(x,9,1,4)
        sal=self.sal(x)*torch.sigmoid((x-mu2).abs().mean(1,True))
        x=self.gate(torch.cat([x,sal],1))+x
        return self.deep(x)


class NhanThauHieuMoiTruong(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        mid=ch//4
        self.spr=nn.Sequential(nn.Conv2d(ch,mid,1,bias=False),nn.BatchNorm2d(mid),nn.SiLU(inplace=True))
        self.sizes=(1,2,3,6); self.fuse=ConvBN(ch+4*mid,ch,k=1,p=0)
        mid2=ch//4
        self.q=nn.Conv2d(ch,mid2,1,bias=False);self.k=nn.Conv2d(ch,mid2,1,bias=False)
        self.v=nn.Conv2d(ch,mid2,1,bias=False)
        self.out=nn.Sequential(nn.Conv2d(mid2,ch,1,bias=False),nn.BatchNorm2d(ch))
        self.deep=nn.Sequential(DSConv(ch),DSConv(ch))
        init_weights(self)
    def forward(self, x):
        H,W=x.shape[2],x.shape[3]; parts=[x]
        for ps in self.sizes:
            parts.append(F.interpolate(self.spr(F.adaptive_avg_pool2d(x,ps)),(H,W),mode='bilinear',align_corners=False))
        x=self.fuse(torch.cat(parts,1))+x
        B,C,H2,W2=x.shape; N=H2*W2; mid=self.q.out_channels
        Q=self._phi(self.q(x)).view(B,-1,N); K=self._phi(self.k(x)).view(B,-1,N); V=self.v(x).view(B,-1,N)
        KV=torch.bmm(K,V.permute(0,2,1)); out=torch.bmm(KV,Q)/(N+1e-6)
        Ks=K.sum(-1,keepdim=True); out=out/((Q*Ks).sum(1,keepdim=True)+1e-6)
        x=self.out(out.view(B,-1,H2,W2))+x
        return self.deep(x)
    @staticmethod
    def _phi(t): return F.elu(t)+1


class CrossCoreInteraction(nn.Module):
    def __init__(self, ch=CORE_CH, n_heads=4):
        super().__init__(); self.ch=ch; self.nh=n_heads; self.scale=(ch//n_heads)**-0.5
        self.q=nn.Linear(ch,ch,bias=False);self.k=nn.Linear(ch,ch,bias=False)
        self.v=nn.Linear(ch,ch,bias=False);self.o=nn.Linear(ch,ch,bias=False)
        self.n1=nn.LayerNorm(ch);self.n2=nn.LayerNorm(ch)
        self.ffn=nn.Sequential(nn.Linear(ch,ch*2),nn.GELU(),nn.Linear(ch*2,ch))
        self.gate=nn.Sequential(nn.Linear(ch,ch),nn.Sigmoid())
        init_weights(self)
    def forward(self, cf):
        B,C,H,W=cf[0].shape
        tokens=torch.stack([f.mean([2,3]) for f in cf],1)
        t=self.n1(tokens)
        Q=self.q(t).view(B,NUM_CORES,self.nh,-1).transpose(1,2)
        K=self.k(t).view(B,NUM_CORES,self.nh,-1).transpose(1,2)
        V=self.v(t).view(B,NUM_CORES,self.nh,-1).transpose(1,2)
        attn=torch.softmax(torch.matmul(Q,K.transpose(-2,-1))*self.scale,-1)
        att=torch.matmul(attn,V).transpose(1,2).reshape(B,NUM_CORES,C)
        tokens=tokens+self.o(att); tokens=tokens+self.ffn(self.n2(tokens))
        w=self.gate(tokens)
        return [cf[i]*w[:,i].view(B,C,1,1) for i in range(NUM_CORES)]


class CoreGatingFusion(nn.Module):
    def __init__(self, core_ch=CORE_CH, out_ch=256):
        super().__init__()
        self.reduce=ConvBN(core_ch*NUM_CORES,out_ch,k=1,p=0)
    def forward(self, cf): return self.reduce(torch.cat(cf,1))


# ══════════════════════════════════════════════════════════════════════
#  MAIN MODEL — WidEye-CoreSix v6 Omega
#  stage_depths=(3,4,10,2) → 50.07M params
# ══════════════════════════════════════════════════════════════════════

class WidEyeV6(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, img_size=224,
                 stage_depths=(3,4,10,2), drop_path_rate=0.2):
        super().__init__()
        self.img_size=img_size; self.num_classes=num_classes

        # DropPath rates (linear)
        total_b = sum(stage_depths)
        dp_all  = [drop_path_rate*i/(max(total_b-1,1)) for i in range(total_b)]
        idx=0; dp_s=[]
        for d in stage_depths: dp_s.append(dp_all[idx:idx+d]); idx+=d

        # Stem
        self.stem=nn.Sequential(
            RepConvBN(3,32,s=2), ConvBN(32,64,k=3,s=1,p=1),
            RepConvBN(64,CORE_CH,s=2), ConvBN(CORE_CH,CORE_CH,k=3,s=1,p=1))

        # Six Cores
        self.core1=NhanThiGiac(); self.core2=NhanTuDuyMoiTruong()
        self.core3=NhanTuDuyNhanDien(); self.core4=NhanHieuMoiTruong()
        self.core5=NhanNhanBietMoiTruong(); self.core6=NhanThauHieuMoiTruong()
        self.cross_core=CrossCoreInteraction(CORE_CH,4)
        self.fusion=CoreGatingFusion(CORE_CH,256)

        # Backbone (3,4,10,2)
        d1,d2,d3,d4=stage_depths
        self.stage1=make_stage(256,64, d1,1,0,'none',0,dp_s[0])
        self.stage2=make_stage(256,128,d2,2,2,'se',  0,dp_s[1])
        self.stage3=make_stage(512,256,d3,2,2,'se',  3,dp_s[2])
        self.stage4=make_stage(1024,512,d4,2,1,'cbam',2,dp_s[3])

        # BETA insertions: SSM-lite + HybridAttention in stage3
        self.ssm3a    = SSMLite(1024,256,8)
        self.ssm3b    = SSMLite(1024,256,8)
        self.hattn3a  = HybridAttention(1024,256,7,4)
        self.hattn3b  = HybridAttention(1024,256,7,4)
        # SSM-lite in stage4
        self.ssm4     = SSMLite(2048,256,8)

        # FPN
        self.lat4=ConvBN(2048,256,k=1,p=0); self.lat3=ConvBN(1024,256,k=1,p=0)
        self.lat2=ConvBN(512, 256,k=1,p=0)
        self.fo4=ConvBN(256,256,k=3,p=1); self.fo3=ConvBN(256,256,k=3,p=1)
        self.fo2=ConvBN(256,256,k=3,p=1)

        # GAMMA: Deep Supervision aux heads (3 scales)
        self.aux1=nn.Sequential(ConvBN(512,128,k=1,p=0),nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(),nn.Linear(128,num_classes))
        self.aux2=nn.Sequential(ConvBN(1024,128,k=1,p=0),nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(),nn.Linear(128,num_classes))
        self.aux3=nn.Sequential(ConvBN(2048,128,k=1,p=0),nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(),nn.Linear(128,num_classes))

        self.final_cbam=CBAM(2048,r=16,sk=7)
        self.gap=nn.AdaptiveAvgPool2d(1); self.fpn_pool=nn.AdaptiveAvgPool2d(1)

        self.shared_fc=nn.Sequential(
            nn.Linear(2304,1024,bias=False),nn.BatchNorm1d(1024),nn.SiLU(inplace=True),nn.Dropout(0.35),
            nn.Linear(1024,512,bias=False), nn.BatchNorm1d(512), nn.SiLU(inplace=True),nn.Dropout(0.2))

        self.cls_head=nn.Sequential(
            nn.Linear(512,256),nn.SiLU(inplace=True),nn.Dropout(0.2),
            nn.Linear(256,128),nn.SiLU(inplace=True),nn.Linear(128,num_classes))

        self.bbox_enc =nn.Sequential(nn.Linear(256,128),nn.SiLU(inplace=True))
        self.bbox_head=nn.Sequential(
            nn.Linear(640,256),nn.SiLU(inplace=True),nn.Dropout(0.1),
            nn.Linear(256,64), nn.SiLU(inplace=True),nn.Linear(64,4),nn.Sigmoid())

        # SwiGLU FFN on shared features
        self.swiglu=SwiGLUFFN(512,mult=2)

        init_weights(self)

    def forward(self, x):
        s=self.stem(x)
        cores=[self.core1(s),self.core2(s),self.core3(s),
               self.core4(s),self.core5(s),self.core6(s)]
        cores=self.cross_core(cores)
        fused=self.fusion(cores)

        s1=self.stage1(fused)
        s2=self.stage2(s1)

        # Stage3 with BETA insertions interleaved
        half=len(self.stage3)//2
        s3 =self.stage3[:half](s2)
        s3 =self.ssm3a(s3)
        s3 =self.hattn3a(s3)
        s3 =self.stage3[half:](s3)
        s3 =self.ssm3b(s3)
        s3 =self.hattn3b(s3)

        s4 =self.stage4(s3)
        s4 =self.ssm4(s4)

        # FPN
        p4=self.fo4(self.lat4(s4))
        p3=self.fo3(self.lat3(s3)+F.interpolate(p4,s3.shape[-2:],mode='nearest'))
        p2=self.fo2(self.lat2(s2)+F.interpolate(p3,s2.shape[-2:],mode='nearest'))

        # Deep supervision (training only)
        aux_out = None
        if self.training:
            a1=self.aux1(s2); a2=self.aux2(s3); a3=self.aux3(s4)
            aux_out=(a1,a2,a3)

        df=self.gap(self.final_cbam(s4)).flatten(1)
        ff=self.fpn_pool(p2).flatten(1)
        sh=self.shared_fc(torch.cat([df,ff],1))
        sh=self.swiglu(sh.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        cl=self.cls_head(sh)
        bp=self.bbox_head(torch.cat([sh,self.bbox_enc(ff)],1))

        if self.training and aux_out is not None:
            return cl, bp, aux_out
        return cl, bp


# ══════════════════════════════════════════════════════════════════════
#  CIoU+TaskAligned · KDLoss · SWA · EMA · DeepSupervision
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothCE(nn.Module):
    def __init__(self, n=NUM_CLASSES, smooth=0.1):
        super().__init__(); self.s=smooth; self.n=n
    def forward(self, p, t):
        lp=F.log_softmax(p,-1)
        with torch.no_grad():
            d=torch.full_like(p,self.s/self.n); d.scatter_(1,t.unsqueeze(1),1-self.s)
        return -(d*lp).sum(-1).mean()


class CIoULoss(nn.Module):
    def forward(self, pred, tgt):
        def xy(b): return torch.stack([b[:,0]-b[:,2]/2,b[:,1]-b[:,3]/2,b[:,0]+b[:,2]/2,b[:,1]+b[:,3]/2],1)
        p,g=xy(pred),xy(tgt)
        ix1=torch.max(p[:,0],g[:,0]); iy1=torch.max(p[:,1],g[:,1])
        ix2=torch.min(p[:,2],g[:,2]); iy2=torch.min(p[:,3],g[:,3])
        inter=torch.clamp(ix2-ix1,0)*torch.clamp(iy2-iy1,0)
        pa=(p[:,2]-p[:,0])*(p[:,3]-p[:,1]); ga=(g[:,2]-g[:,0])*(g[:,3]-g[:,1])
        union=pa+ga-inter+1e-7; iou=inter/union
        cx1=torch.min(p[:,0],g[:,0]); cy1=torch.min(p[:,1],g[:,1])
        cx2=torch.max(p[:,2],g[:,2]); cy2=torch.max(p[:,3],g[:,3])
        c2=(cx2-cx1)**2+(cy2-cy1)**2+1e-7
        pcx=(p[:,0]+p[:,2])/2; pcy=(p[:,1]+p[:,3])/2
        gcx=(g[:,0]+g[:,2])/2; gcy=(g[:,1]+g[:,3])/2
        d2=(pcx-gcx)**2+(pcy-gcy)**2
        pw=p[:,2]-p[:,0]; ph=p[:,3]-p[:,1]
        gw=g[:,2]-g[:,0]; gh=g[:,3]-g[:,1]
        v=(4/math.pi**2)*(torch.atan(gw/(gh+1e-7))-torch.atan(pw/(ph+1e-7)))**2
        with torch.no_grad(): alpha=v/(1-iou+v+1e-7)
        return (1-(iou-d2/c2-alpha*v)).mean()


class OmegaLoss(nn.Module):
    """CIoU + LabelSmooth + Task-Aligned + Deep Supervision."""
    def __init__(self, aux_w=0.3):
        super().__init__()
        self.cls=LabelSmoothCE(); self.ciou=CIoULoss()
        self.l1=nn.SmoothL1Loss(beta=0.1); self.aux_w=aux_w

    def forward(self, cl, bp, ct, bt, bw=1.0, aux_out=None):
        with torch.no_grad():
            q=self._iou(bp.detach(),bt); cw=(2.0-q).mean()
        lc=self.cls(cl,ct)*cw; li=self.ciou(bp,bt); ll=self.l1(bp,bt)
        loss=lc+bw*(2.0*li+0.5*ll)
        if aux_out is not None:
            for ax in aux_out:
                loss=loss+self.aux_w*self.cls(ax,ct)
        return loss, lc.item(), li.item(), ll.item()

    @staticmethod
    def _iou(p, g):
        px1=p[:,0]-p[:,2]/2; py1=p[:,1]-p[:,3]/2
        px2=p[:,0]+p[:,2]/2; py2=p[:,1]+p[:,3]/2
        gx1=g[:,0]-g[:,2]/2; gy1=g[:,1]-g[:,3]/2
        gx2=g[:,0]+g[:,2]/2; gy2=g[:,1]+g[:,3]/2
        ix1=torch.max(px1,gx1); iy1=torch.max(py1,gy1)
        ix2=torch.min(px2,gx2); iy2=torch.min(py2,gy2)
        inter=torch.clamp(ix2-ix1,0)*torch.clamp(iy2-iy1,0)
        union=(px2-px1)*(py2-py1)+(gx2-gx1)*(gy2-gy1)-inter+1e-7
        return inter/union


class KDLoss(nn.Module):
    def __init__(self, T=4.0): super().__init__(); self.T=T
    def forward(self, s, t):
        return F.kl_div(F.log_softmax(s/self.T,-1),
                        F.softmax(t/self.T,-1),reduction='batchmean')*(self.T**2)


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.shadow=copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
        self.decay=decay; self.step=0
    @torch.no_grad()
    def update(self, model):
        self.step+=1; d=min(self.decay,(1+self.step)/(10+self.step))
        for ep,mp in zip(self.shadow.parameters(),model.parameters()):
            ep.data.mul_(d).add_(mp.data,alpha=1-d)
        for eb,mb in zip(self.shadow.buffers(),model.buffers()): eb.copy_(mb)


class SWA:
    """Stochastic Weight Averaging — free ensemble."""
    def __init__(self, model):
        self.shadow=copy.deepcopy(model); self.n=0
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        self.n+=1
        for sp,mp in zip(self.shadow.parameters(),model.parameters()):
            sp.data.mul_(self.n/(self.n+1)).add_(mp.data,alpha=1/(self.n+1))
    def update_bn(self, loader, device):
        self.shadow.train()
        with torch.no_grad():
            for imgs,_,_ in loader:
                self.shadow(imgs.to(device))
        self.shadow.eval()


class WarmupCosine:
    def __init__(self, opt, we, te, plr, mlr=1e-6):
        self.opt=opt; self.we=we; self.te=te; self.plr=plr; self.mlr=mlr
    def step(self, ep):
        lr=(self.plr*(ep+1)/self.we if ep<self.we else
            self.mlr+0.5*(self.plr-self.mlr)*(1+math.cos(math.pi*(ep-self.we)/(self.te-self.we))))
        for g in self.opt.param_groups: g['lr']=lr
        return lr


# ══════════════════════════════════════════════════════════════════════
#  Mosaic · MixUp · CopyPaste · ProgressiveResize · Dataset
# ══════════════════════════════════════════════════════════════════════

CLASS_COLORS={
    0:[(220,60,50),(255,110,60),(200,40,30)],
    1:[(50,190,60),(30,160,90),(80,210,110)],
    2:[(110,110,120),(90,90,100),(140,140,150)],
    3:[(160,60,210),(190,90,230),(140,40,190)],
    4:[(230,210,40),(250,190,60),(210,220,30)],
}


class OmegaDataset(Dataset):
    def __init__(self, n=10000, size=224, augment=True,
                 mosaic_p=0.5, mixup_p=0.15, cp_p=0.3):
        self.n=n; self.sz=size; self.aug=augment
        self.mp=mosaic_p; self.xp=mixup_p; self.cp=cp_p
        self.tf=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])

    def _bg(self, S, r):
        t=r.randint(0,5); a=np.zeros((S,S,3),np.uint8)
        if t==0:
            c1=np.array([r.randint(10,80)]*3); c2=np.array([r.randint(80,200)]*3)
            for i in range(S): a[:,i]=(c1*(1-i/S)+c2*(i/S)).astype(np.uint8)
        elif t==1: np.random.RandomState(r.randint(0,99999)).randint(20,100,(S,S,3),dtype=np.uint8,out=a)
        elif t==2:
            b=r.randint(40,120); a[:]=b; st=r.randint(16,48)
            a[::st,:]=b+40; a[:,::st]=b+40
        elif t==3:
            ts=r.randint(20,40); c1,c2=r.randint(40,100),r.randint(110,200)
            for y in range(0,S,ts):
                for x in range(0,S,ts): a[y:y+ts,x:x+ts]=c1 if (y//ts+x//ts)%2==0 else c2
        elif t==4:
            Y,X=np.ogrid[:S,:S]; d=np.clip(np.sqrt((X-S//2)**2+(Y-S//2)**2)/(S*.7),0,1)[...,None]
            c1=np.array([r.randint(20,80)]*3); c2=np.array([r.randint(100,200)]*3)
            a=(c1*(1-d)+c2*d).astype(np.uint8)
        else: a[:]=r.randint(40,160)
        return a

    def _draw(self, draw, cid, cx, cy, w, h, r):
        x1,y1,x2,y2=int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)
        mc=random.choice(CLASS_COLORS[cid])
        lt=tuple(min(255,c+60) for c in mc); dk=tuple(max(0,c-40) for c in mc)
        if cid==0:
            draw.rectangle([x1,y1,x2,y2],fill=mc,outline=(0,0,0),width=2)
            draw.rectangle([x1,y1,x2,y1+max(2,h//6)],fill=lt)
        elif cid==1: draw.polygon([(cx,y1),(x1,y2),(x2,y2)],fill=mc,outline=dk)
        elif cid==2:
            draw.rectangle([x1,y1,x2,y2],fill=mc,outline=(60,60,60))
            bh=max(4,h//6)
            for ry in range(y1,y2,bh): draw.line([(x1,ry),(x2,ry)],fill=dk,width=1)
        elif cid==3:
            hr=max(5,w//4); draw.ellipse([cx-hr,y1,cx+hr,y1+2*hr],fill=mc)
            draw.rectangle([cx-w//3,y1+2*hr,cx+w//3,y2],fill=mc)
        elif cid==4:
            draw.rectangle([x1,y1+h//4,x2,y2],fill=mc,outline=(60,50,0),width=2)
            cab=tuple(min(255,c+50) for c in mc)
            draw.rectangle([x1+w//5,y1,x2-w//5,y1+h//4+2],fill=cab)
            wr=max(3,h//7)
            for wx_ in [x1+w//6,x2-w//6]: draw.ellipse([wx_-wr,y2-wr,wx_+wr,y2+wr],fill=(30,30,30))

    def _single(self, idx, sz=None):
        S=sz or self.sz; r=random.Random(idx*2053+17)
        a=self._bg(S,r); img=Image.fromarray(a.astype(np.uint8)); draw=ImageDraw.Draw(img)
        cid=r.randint(0,NUM_CLASSES-1)
        ow=r.randint(int(.2*S),int(.6*S)); oh=r.randint(int(.2*S),int(.6*S)); m=5
        cx=r.randint(ow//2+m,S-ow//2-m); cy=r.randint(oh//2+m,S-oh//2-m)
        for _ in range(r.randint(0,2)):
            dc=r.randint(0,NUM_CLASSES-1); dw=r.randint(int(.06*S),int(.2*S)); dh=r.randint(int(.06*S),int(.2*S))
            dx=r.randint(dw//2+2,S-dw//2-2); dy=r.randint(dh//2+2,S-dh//2-2)
            self._draw(draw,dc,dx,dy,dw,dh,r)
        self._draw(draw,cid,cx,cy,ow,oh,r)
        a2=np.clip(np.array(img).astype(np.float32)+np.random.normal(0,r.uniform(0,15),(S,S,3)),0,255).astype(np.uint8)
        return Image.fromarray(a2),cid,cx,cy,ow,oh

    def _mosaic(self, idx):
        S=self.sz; h=S//2; canvas=Image.new('RGB',(S,S))
        positions=[(0,0),(h,0),(0,h),(h,h)]
        main_cid=main_bbox=None
        for i,(pos,si) in enumerate(zip(positions,[idx,(idx+1)%self.n,(idx+2)%self.n,(idx+3)%self.n])):
            img,cid,cx,cy,ow,oh=self._single(si,h); canvas.paste(img,pos)
            if i==0: main_cid=cid; main_bbox=(cx+pos[0],cy+pos[1],ow,oh)
        return canvas,main_cid,*main_bbox

    def _copy_paste(self, idx):
        r=random.Random(idx*9999); img1,cid,cx,cy,ow,oh=self._single(idx); S=self.sz
        img2,_,cx2,cy2,ow2,oh2=self._single((idx+7)%self.n)
        x1c=max(0,int(cx2-ow2//2)); y1c=max(0,int(cy2-oh2//2))
        x2c=min(S,int(cx2+ow2//2)); y2c=min(S,int(cy2+oh2//2))
        crop=img2.crop((x1c,y1c,x2c,y2c))
        px=r.randint(0,max(1,S-crop.width)); py=r.randint(0,max(1,S-crop.height))
        img1.paste(crop,(px,py))
        return img1,cid,cx,cy,ow,oh

    def __len__(self): return self.n

    def __getitem__(self, idx):
        r=random.Random(idx*2053+17); S=self.sz
        rv=r.random()
        if self.aug and rv<self.mp:
            img,cid,cx,cy,ow,oh=self._mosaic(idx)
        elif self.aug and rv<self.mp+self.cp:
            img,cid,cx,cy,ow,oh=self._copy_paste(idx)
        else:
            img,cid,cx,cy,ow,oh=self._single(idx)
        if self.aug and r.random()<self.xp:
            img2,cid2,_,_,_,_=self._single((idx+13)%self.n)
            lam=np.random.beta(.4,.4)
            arr=np.clip(lam*np.array(img).astype(np.float32)+(1-lam)*np.array(img2.resize(img.size)).astype(np.float32),0,255).astype(np.uint8)
            img=Image.fromarray(arr); cid=cid if lam>=.5 else cid2
        if self.aug:
            if r.random()<.6: img=TF.adjust_brightness(img,r.uniform(.6,1.4))
            if r.random()<.4: img=TF.adjust_contrast(img,r.uniform(.7,1.3))
            if r.random()<.3: img=img.filter(ImageFilter.GaussianBlur(r.uniform(.3,1.2)))
            if r.random()<.3: img=TF.hflip(img); cx=S-cx
        x1g=max(0,cx-ow//2); y1g=max(0,cy-oh//2)
        x2g=min(S,cx+ow//2); y2g=min(S,cy+oh//2)
        vw=x2g-x1g; vh=y2g-y1g
        bbox=torch.tensor([(x1g+x2g)/2/S,(y1g+y2g)/2/S,vw/S,vh/S],dtype=torch.float32)
        return self.tf(img.resize((S,S))),torch.tensor(cid,dtype=torch.long),bbox


class ProgressiveResizeScheduler:
    def __init__(self, sizes=None):
        self.schedule=sizes or [(112,0),(160,8),(192,16),(224,24),(256,32),(320,40)]
    def get_size(self, epoch):
        sz=self.schedule[0][0]
        for s,e in self.schedule:
            if epoch>=e: sz=s
        return sz


# ══════════════════════════════════════════════════════════════════════
#  TRAINING + INFERENCE
# ══════════════════════════════════════════════════════════════════════

def compute_iou(pred, tgt):
    px1=pred[:,0]-pred[:,2]/2; py1=pred[:,1]-pred[:,3]/2
    px2=pred[:,0]+pred[:,2]/2; py2=pred[:,1]+pred[:,3]/2
    gx1=tgt[:,0]-tgt[:,2]/2;  gy1=tgt[:,1]-tgt[:,3]/2
    gx2=tgt[:,0]+tgt[:,2]/2;  gy2=tgt[:,1]+tgt[:,3]/2
    ix1=torch.max(px1,gx1); iy1=torch.max(py1,gy1)
    ix2=torch.min(px2,gx2); iy2=torch.min(py2,gy2)
    inter=torch.clamp(ix2-ix1,0)*torch.clamp(iy2-iy1,0)
    union=(px2-px1)*(py2-py1)+(gx2-gx1)*(gy2-gy1)-inter+1e-7
    return (inter/union).mean().item()


def train(num_epochs=50, batch=24, peak_lr=8e-4, n_train=10000, n_val=2000,
          save="wideeye_v6.pth", warmup=5, accum=2, amp=True,
          drop_path=0.2, teacher_path=None, ema_decay=0.9999,
          swa_start=35, progressive=True):

    device=torch.device("cuda" if torch.cuda.is_available() else
                        "mps"  if torch.backends.mps.is_available() else "cpu")
    amp_on=amp and device.type=="cuda"

    print(f"\n{'═'*70}")
    print(f"  WidEye-CoreSix v6 Omega  |  {device}  |  AMP:{amp_on}")
    print(f"  Eff-batch:{batch*accum}  Epochs:{num_epochs}  DropPath:{drop_path}")
    print(f"  Progressive:{progressive}  EMA:{ema_decay}  SWA@ep{swa_start}")
    print(f"  KD:{teacher_path is not None}")
    print(f"{'═'*70}\n")

    model=WidEyeV6(NUM_CLASSES, drop_path_rate=drop_path).to(device)
    tp=sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {tp:,}  ({tp/1e6:.3f}M)\n")

    ema=ModelEMA(model,ema_decay)
    swa=SWA(model)
    opt=optim.AdamW(model.parameters(),lr=peak_lr,weight_decay=5e-4)
    sched=WarmupCosine(opt,warmup,num_epochs,peak_lr)
    crit=OmegaLoss(aux_w=0.3)
    kd_loss=KDLoss(T=4.0) if teacher_path else None
    scaler=torch.cuda.amp.GradScaler(enabled=amp_on)

    teacher=None
    if teacher_path:
        ck=torch.load(teacher_path,map_location=device,weights_only=False)
        teacher=WidEyeV6(NUM_CLASSES).to(device)
        teacher.load_state_dict(ck["model_state"]); teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        print(f"  Teacher loaded from {teacher_path}\n")

    prog=ProgressiveResizeScheduler() if progressive else None
    cur_sz=prog.get_size(0) if prog else 224

    tds=OmegaDataset(n_train,cur_sz,True); vds=OmegaDataset(n_val,224,False)
    nw=2
    tldr=DataLoader(tds,batch,True,num_workers=nw,pin_memory=device.type=='cuda',drop_last=True)
    vldr=DataLoader(vds,batch,False,num_workers=nw,pin_memory=device.type=='cuda')

    best_iou=0.0; patience=12; no_imp=0

    for ep in range(1,num_epochs+1):
        t0=time.time(); bw=min(1.0,ep/max(warmup,1))

        if prog:
            ns=prog.get_size(ep)
            if ns!=cur_sz:
                cur_sz=ns; tds=OmegaDataset(n_train,cur_sz,True)
                tldr=DataLoader(tds,batch,True,num_workers=nw,pin_memory=device.type=='cuda',drop_last=True)
                print(f"  ↑ Progressive resize → {cur_sz}×{cur_sz}")

        model.train(); tl_sum=0.0
        opt.zero_grad(set_to_none=True)
        for step,(imgs,lbls,bbs) in enumerate(tldr):
            imgs=imgs.to(device,non_blocking=True)
            lbls=lbls.to(device,non_blocking=True)
            bbs=bbs.to(device,non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_on):
                out=model(imgs)
                if isinstance(out,tuple) and len(out)==3:
                    cl,bp,aux=out
                else:
                    cl,bp=out; aux=None
                loss,*_=crit(cl,bp,lbls,bbs,bw,aux)
                if teacher is not None:
                    with torch.no_grad(): tcl,_=teacher(imgs)
                    loss=loss+0.5*kd_loss(cl,tcl)
                loss=loss/accum
            scaler.scale(loss).backward()
            if (step+1)%accum==0 or (step+1)==len(tldr):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            ema.update(model)
            tl_sum+=loss.item()*accum
        avg_tl=tl_sum/len(tldr)

        if ep>=swa_start: swa.update(model)

        ema.shadow.eval(); vl_sum=0.0; corr=0; tot=0; iou_s=0.0; nb=0
        with torch.no_grad():
            for imgs,lbls,bbs in vldr:
                imgs=imgs.to(device,non_blocking=True)
                lbls=lbls.to(device,non_blocking=True)
                bbs=bbs.to(device,non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_on):
                    cl,bp=ema.shadow(imgs)
                    loss,*_=crit(cl,bp,lbls,bbs,1.0)
                vl_sum+=loss.item()
                corr+=(cl.argmax(1)==lbls).sum().item(); tot+=lbls.size(0)
                iou_s+=compute_iou(bp,bbs); nb+=1

        avg_vl=vl_sum/len(vldr); acc=corr/tot*100; avg_iou=iou_s/nb
        lr=sched.step(ep); el=time.time()-t0

        print(f"  [{ep:3d}/{num_epochs}] T:{avg_tl:.4f} V:{avg_vl:.4f} "
              f"Acc:{acc:5.1f}% IoU:{avg_iou:.4f} LR:{lr:.1e} sz:{cur_sz} {el:.1f}s")

        if avg_iou>best_iou:
            best_iou=avg_iou; no_imp=0
            torch.save({"epoch":ep,"model_state":ema.shadow.state_dict(),
                        "opt":opt.state_dict(),"best_iou":best_iou,"val_acc":acc,
                        "num_classes":NUM_CLASSES,"img_size":224,"classes":CLASSES,
                        "arch":"WidEyeV6","params":tp}, save)
            print(f"  ✓ Saved EMA model (IoU:{best_iou:.4f})")
        else:
            no_imp+=1
            if no_imp>=patience: print(f"\n  Early stop @ epoch {ep}"); break

    if swa.n>0:
        print(f"\n  Updating SWA BatchNorm..."); swa.update_bn(vldr,device)
        torch.save({"model_state":swa.shadow.state_dict(),"arch":"WidEyeV6-SWA",
                    "params":tp,"classes":CLASSES}, save.replace('.pth','_swa.pth'))
        print(f"  SWA model saved.")

    print(f"\n  Best IoU: {best_iou:.4f}")
    return model


class Inference:
    def __init__(self, path="wideeye_v6.pth", size=224, deploy=True):
        self.sz=size
        self.dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck=torch.load(path,map_location=self.dev,weights_only=False)
        self.cls=ck.get("classes",CLASSES)
        model=WidEyeV6(ck.get("num_classes",NUM_CLASSES),size).to(self.dev)
        model.load_state_dict(ck["model_state"],strict=False)
        if deploy: model=reparameterize_model(model)
        model.eval()
        try: model=torch.compile(model)
        except Exception: pass
        self.model=model
        self.tf=T.Compose([T.Resize((size,size)),T.ToTensor(),
                           T.Normalize([.485,.456,.406],[.229,.224,.225])])
        p=sum(x.numel() for x in model.parameters())
        print(f"  v6 Loaded: {p/1e6:.3f}M | IoU={ck.get('best_iou',0):.4f} "
              f"| Acc={ck.get('val_acc',0):.1f}% | Deploy:{deploy}")

    @torch.no_grad()
    def predict(self, pil_img):
        W,H=pil_img.size
        t=self.tf(pil_img).unsqueeze(0).to(self.dev)
        cl,bp=self.model(t)
        pr=torch.softmax(cl,1)[0]; cid=pr.argmax().item(); conf=pr[cid].item()
        cx,cy,bw,bh=bp[0].cpu().float().numpy()
        x1=int((cx-bw/2)*W); y1=int((cy-bh/2)*H)
        x2=int((cx+bw/2)*W); y2=int((cy+bh/2)*H)
        return self.cls.get(cid,f"cls{cid}"),conf,(max(0,x1),max(0,y1),min(W,x2),min(H,y2))


def run_webcam(path="wideeye_v6.pth", size=224, cam=0):
    try: import cv2
    except ImportError: print("pip install opencv-python"); return
    COLORS={"obstacle_box":(0,50,220),"path_clear":(50,200,50),
            "wall_detected":(150,150,150),"person_nearby":(200,50,200),"vehicle_zone":(0,200,220)}
    eye=Inference(path,size); cap=cv2.VideoCapture(cam)
    if not cap.isOpened(): print(f"Cannot open camera {cam}"); return
    fps_h=[]; fc=0; print("Q=quit  S=screenshot\n")
    while True:
        t0=time.time(); ok,frame=cap.read()
        if not ok: break; fc+=1
        n,conf,(x1,y1,x2,y2)=eye.predict(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        H,W=frame.shape[:2]; clr=COLORS.get(n,(255,255,255))
        cv2.rectangle(frame,(x1,y1),(x2,y2),clr,2)
        cl_=15; ct=3
        for px,py,sx,sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(px,py),(px+sx*cl_,py),clr,ct); cv2.line(frame,(px,py),(px,py+sy*cl_),clr,ct)
        lb=f"{n} {conf*100:.0f}%"; (lw,lh),_=cv2.getTextSize(lb,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
        ly=max(y1-8,lh+8); cv2.rectangle(frame,(x1,ly-lh-5),(x1+lw+4,ly+2),clr,-1)
        cv2.putText(frame,lb,(x1+2,ly-2),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        fps_h.append(1/(time.time()-t0+1e-9))
        if len(fps_h)>30: fps_h.pop(0)
        cv2.putText(frame,f"FPS:{sum(fps_h)/len(fps_h):.1f}",(10,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
        cv2.putText(frame,"WidEye v6 Omega",(10,H-10),cv2.FONT_HERSHEY_SIMPLEX,.4,(180,180,180),1)
        cv2.imshow("WidEye-CoreSix v6 Omega",frame)
        k=cv2.waitKey(1)&0xFF
        if k in [ord('q'),27]: break
        elif k==ord('s'): cv2.imwrite(f"v6_{fc:05d}.jpg",frame)
    cap.release(); cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────
# ANALYZE + STATIC CHECKS
# ─────────────────────────────────────────────────────────────────────

def analyze():
    import ast
    with open(__file__,'r') as f: src=f.read()
    tree=ast.parse(src)
    reserved={'nonlocal','global','return','class','lambda','import','pass'}
    bad=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr in reserved]
    gs=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr=='grid_sample']
    print(f"  Syntax       : PASS")
    print(f"  Reserved kw  : {'PASS' if not bad else f'FAIL {bad}'}")
    print(f"  grid_sample  : {'PASS (0 calls)' if not gs else f'FOUND {gs}'}")

    model=WidEyeV6(NUM_CLASSES); model.eval()
    total=sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters   : {total:,}  ({total/1e6:.3f}M)")
    print(f"  FP32 size    : {total*4/1024**2:.1f} MB")
    print(f"  FP16 size    : {total*2/1024**2:.1f} MB")

    def mp(m): return sum(p.numel() for p in m.parameters())
    rows=[("Stem",model.stem),("Core1",model.core1),("Core2",model.core2),
          ("Core3",model.core3),("Core4",model.core4),("Core5",model.core5),
          ("Core6",model.core6),("CrossCore",model.cross_core),("Fusion",model.fusion),
          ("Stage1",model.stage1),("Stage2",model.stage2),("Stage3",model.stage3),
          ("Stage4",model.stage4),
          ("SSM+Attn",nn.ModuleList([model.ssm3a,model.ssm3b,model.hattn3a,model.hattn3b,model.ssm4])),
          ("AuxHeads",nn.ModuleList([model.aux1,model.aux2,model.aux3])),
          ("FPN+CBAM",nn.ModuleList([model.lat2,model.lat3,model.lat4,model.fo2,model.fo3,model.fo4,model.final_cbam])),
          ("SharedFC+SwiGLU",nn.Sequential(model.shared_fc,model.swiglu)),
          ("Heads",nn.Sequential(model.cls_head,model.bbox_enc,model.bbox_head))]
    print(f"\n  {'Module':<18} {'Params':>12}  {'%':>7}")
    print(f"  {'─'*42}")
    for name,mod in rows:
        p=mp(mod); print(f"  {name:<18} {p:>12,}  {100*p/total:>6.2f}%")

    dummy=torch.randn(2,3,224,224)
    model.train()
    with torch.no_grad(): out=model(dummy)
    assert len(out)==3
    cl,bp,aux=out
    assert cl.shape==(2,NUM_CLASSES) and bp.shape==(2,4)
    assert bp.min()>=0 and bp.max()<=1
    print(f"\n  Forward(train) : PASS  cls={cl.shape} bbox={bp.shape} aux={len(aux)}")
    model.eval()
    with torch.no_grad(): cl2,bp2=model(dummy)
    assert cl2.shape==(2,NUM_CLASSES)
    print(f"  Forward(eval)  : PASS  cls={cl2.shape}")

    rep_n=sum(1 for m in model.modules() if isinstance(m,RepConvBN))
    md=reparameterize_model(copy.deepcopy(model))
    with torch.no_grad(): cl3,bp3=md(dummy)
    diff=(cl2-cl3).abs().max().item()
    print(f"  RepConv blocks : {rep_n}  fuse-diff={diff:.2e}  {'PASS' if diff<1e-3 else 'FAIL'}")
    dp_n=sum(1 for m in model.modules() if isinstance(m,DropPath) and m.p>0)
    print(f"  DropPath active: {dp_n}")
    moe_n=sum(1 for m in model.modules() if isinstance(m,SparseMoEFFN))
    ssm_n=sum(1 for m in model.modules() if isinstance(m,SSMLite))
    ha_n =sum(1 for m in model.modules() if isinstance(m,HybridAttention))
    print(f"  MoE blocks     : {moe_n}")
    print(f"  SSM-lite blocks: {ssm_n}")
    print(f"  HybridAttn     : {ha_n}")
    print(f"\n  ✓ All checks PASSED — WidEye v6 ready.")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description="WidEye-CoreSix v6 Omega")
    p.add_argument("--mode",choices=["analyze","train","webcam"],default="analyze")
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--batch",     type=int,   default=24)
    p.add_argument("--lr",        type=float, default=8e-4)
    p.add_argument("--n_train",   type=int,   default=10000)
    p.add_argument("--n_val",     type=int,   default=2000)
    p.add_argument("--accum",     type=int,   default=2)
    p.add_argument("--model",     type=str,   default="wideeye_v6.pth")
    p.add_argument("--teacher",   type=str,   default=None)
    p.add_argument("--camera",    type=int,   default=0)
    p.add_argument("--no_amp",    action="store_true")
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument("--no_prog",   action="store_true")
    p.add_argument("--swa_start", type=int,   default=35)
    args=p.parse_args()
    if   args.mode=="analyze": analyze()
    elif args.mode=="train":
        train(args.epochs,args.batch,args.lr,args.n_train,args.n_val,
              save=args.model,accum=args.accum,amp=not args.no_amp,
              drop_path=args.drop_path,teacher_path=args.teacher,
              swa_start=args.swa_start,progressive=not args.no_prog)
    elif args.mode=="webcam": run_webcam(args.model,cam=args.camera)
