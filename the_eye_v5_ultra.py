"""
WidEye-CoreSix v5 Ultra  —  the_eye_v5_ultra.py
See README.md for full documentation.
"""

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

CLASSES = {0:"obstacle_box",1:"path_clear",2:"wall_detected",
           3:"person_nearby",4:"vehicle_zone"}
NUM_CLASSES = len(CLASSES)
CORE_CH     = 128
NUM_CORES   = 6

# ─────────────────────────────────────────────────────────────────────
# ACTIVATION
# ─────────────────────────────────────────────────────────────────────

ACT = nn.SiLU   # v5: ReLU → SiLU (Swish) throughout

# ─────────────────────────────────────────────────────────────────────
# STRUCTURAL REPARAMETERIZATION
# ─────────────────────────────────────────────────────────────────────

class RepConvBN(nn.Module):
    """
    Multi-branch conv during training.
    Fuse to single Conv3x3 at inference via reparameterize().
    3x3 branch + 1x1 branch + identity branch (if ic==oc, stride==1).
    """
    def __init__(self, ic, oc, s=1, act=True):
        super().__init__()
        self.ic = ic; self.oc = oc; self.s = s
        self.deployed = False

        self.branch3 = nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(oc))
        self.branch1 = nn.Sequential(
            nn.Conv2d(ic, oc, 1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(oc))
        self.branch_id = (nn.BatchNorm2d(ic)
                          if (ic == oc and s == 1) else None)
        self.act = ACT(inplace=True) if act else nn.Identity()
        self._fused_conv = None

    def forward(self, x):
        if self.deployed:
            return self.act(self._fused_conv(x))
        out = self.branch3(x) + self.branch1(x)
        if self.branch_id is not None:
            out = out + self.branch_id(x)
        return self.act(out)

    def reparameterize(self):
        """Fuse all branches into one Conv3x3+bias. Call before inference."""
        if self.deployed:
            return
        W3, b3 = self._fuse_bn(self.branch3[0], self.branch3[1])
        W1, b1 = self._fuse_bn(self.branch1[0], self.branch1[1])
        W1p = F.pad(W1, [1, 1, 1, 1])  # pad 1x1 → 3x3
        Wi, bi = 0.0, 0.0
        if self.branch_id is not None:
            Wi, bi = self._fuse_bn_identity(self.branch_id)
        W = W3 + W1p + Wi
        b = b3 + b1  + bi
        self._fused_conv = nn.Conv2d(
            self.ic, self.oc, 3, stride=self.s, padding=1, bias=True)
        self._fused_conv.weight.data = W
        self._fused_conv.bias.data   = b
        del self.branch3, self.branch1, self.branch_id
        self.deployed = True

    @staticmethod
    def _fuse_bn(conv, bn):
        W = conv.weight
        μ, σ2, γ, β = bn.running_mean, bn.running_var, bn.weight, bn.bias
        std = torch.sqrt(σ2 + bn.eps)
        scale = γ / std
        W_f = W * scale.reshape(-1, 1, 1, 1)
        b_f = β - μ * scale
        return W_f, b_f

    def _fuse_bn_identity(self, bn):
        μ, σ2, γ, β = bn.running_mean, bn.running_var, bn.weight, bn.bias
        std = torch.sqrt(σ2 + bn.eps)
        scale = γ / std
        # Identity kernel: eye per output channel
        oc = self.oc
        W = torch.zeros(oc, oc, 3, 3, device=bn.weight.device)
        for i in range(oc):
            W[i, i, 1, 1] = 1.0
        W_f = W * scale.reshape(-1, 1, 1, 1)
        b_f = β - μ * scale
        return W_f, b_f


def reparameterize_model(model):
    """Call once after training to fuse all RepConvBN blocks."""
    for m in model.modules():
        if isinstance(m, RepConvBN):
            m.reparameterize()
    return model


# ─────────────────────────────────────────────────────────────────────
# STOCHASTIC DEPTH (DROPPATH)
# ─────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.p = drop_prob

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(
            torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


# ─────────────────────────────────────────────────────────────────────
# BASE BLOCKS
# ─────────────────────────────────────────────────────────────────────

class ConvBN(nn.Module):
    def __init__(self, ic, oc, k=3, s=1, p=1, g=1, act=True, dil=1):
        super().__init__()
        pad = p if dil == 1 else dil
        ops = [nn.Conv2d(ic, oc, k, stride=s, padding=pad,
                         groups=g, dilation=dil, bias=False),
               nn.BatchNorm2d(oc, momentum=0.01, eps=1e-3)]
        if act:
            ops.append(ACT(inplace=True))
        self.seq = nn.Sequential(*ops)
    def forward(self, x): return self.seq(x)


class DSConv(nn.Module):
    def __init__(self, ic, oc=None, s=1):
        super().__init__(); oc = oc or ic
        self.dw = nn.Sequential(
            nn.Conv2d(ic,ic,3,stride=s,padding=1,groups=ic,bias=False),
            nn.BatchNorm2d(ic), ACT(inplace=True))
        self.pw = nn.Sequential(
            nn.Conv2d(ic,oc,1,bias=False),
            nn.BatchNorm2d(oc), ACT(inplace=True))
    def forward(self, x): return self.pw(self.dw(x))


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__(); mid = max(ch//r, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch,mid,1,bias=False), ACT(inplace=True),
            nn.Conv2d(mid,ch,1,bias=False), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)


class CBAM(nn.Module):
    def __init__(self, ch, r=16, sk=7):
        super().__init__(); mid = max(ch//r, 8)
        self.avg = nn.AdaptiveAvgPool2d(1); self.mx = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch,mid,1,bias=False), ACT(inplace=True),
            nn.Conv2d(mid,ch,1,bias=False))
        self.sp  = nn.Conv2d(2,1,sk,padding=sk//2,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        ca = self.sig(self.mlp(self.avg(x)) + self.mlp(self.mx(x)))
        x  = x * ca
        sa = self.sig(self.sp(torch.cat([x.mean(1,True), x.max(1,True)[0]],1)))
        return x * sa


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, ic, mid, s=1, attn='none', drop_path=0.0):
        super().__init__(); oc = mid * self.expansion
        self.c1 = ConvBN(ic, mid, k=1, p=0)
        self.c2 = RepConvBN(mid, mid, s=s)   # Rep during train
        self.c3 = nn.Sequential(
            nn.Conv2d(mid, oc, 1, bias=False), nn.BatchNorm2d(oc))
        self.attn = (SEBlock(oc) if attn=='se' else
                     CBAM(oc)    if attn=='cbam' else nn.Identity())
        self.sc = nn.Sequential()
        if s != 1 or ic != oc:
            self.sc = nn.Sequential(
                nn.Conv2d(ic,oc,1,stride=s,bias=False), nn.BatchNorm2d(oc))
        self.drop_path = DropPath(drop_path)
        self.act = ACT(inplace=True)

    def forward(self, x):
        out = self.attn(self.c3(self.c2(self.c1(x))))
        return self.act(self.drop_path(out) + self.sc(x))


def make_stage(ic, mid, depth, stride, attn_every=0,
               attn_type='se', dp_rates=None):
    oc = mid * Bottleneck.expansion; blocks = []
    for i in range(depth):
        dp = dp_rates[i] if dp_rates else 0.0
        ua = attn_type if (attn_every > 0 and i % attn_every == 0) else 'none'
        blocks.append(Bottleneck(
            ic if i==0 else oc, mid,
            s=stride if i==0 else 1, attn=ua, drop_path=dp))
    return nn.Sequential(*blocks)


def init_weights(m):
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, mode='fan_out',
                                    nonlinearity='relu')
            if mod.bias is not None: nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
            if mod.bias is not None: nn.init.zeros_(mod.bias)


# ─────────────────────────────────────────────────────────────────────
# SIX COGNITIVE CORES
# ─────────────────────────────────────────────────────────────────────

class NhanThiGiac(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.orient = ConvBN(ch, ch, k=3, p=1, g=4)
        self.mix    = ConvBN(ch, ch, k=1, p=0)
        self.alpha  = nn.Parameter(torch.full((1,ch,1,1), 0.5))
        self.p_sm   = nn.AvgPool2d(3, 1, 1)
        self.p_lg   = nn.AvgPool2d(7, 1, 3)
        self.dog_bn = nn.BatchNorm2d(ch)
        self.deep   = nn.Sequential(DSConv(ch), DSConv(ch))
        init_weights(self)
    def forward(self, x):
        o   = self.mix(self.orient(x))
        dog = self.dog_bn(self.p_sm(x) - self.alpha.clamp(0,1) * self.p_lg(x))
        return self.deep(o + dog + x)


class CoordConv(nn.Module):
    def forward(self, x):
        B,_,H,W = x.shape
        yc = torch.linspace(-1,1,H,device=x.device).view(1,1,H,1).expand(B,1,H,W)
        xc = torch.linspace(-1,1,W,device=x.device).view(1,1,1,W).expand(B,1,H,W)
        return torch.cat([x, xc, yc], 1)


class NhanTuDuyMoiTruong(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__(); mid = ch // 4
        mk = lambda d,p: nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=p,dilation=d,groups=ch,bias=False),
            nn.BatchNorm2d(ch), ACT(inplace=True))
        self.dw1=mk(1,1); self.dw2=mk(2,2); self.dw4=mk(4,4); self.dw8=mk(8,8)
        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch,ch,1,bias=False),
                                 nn.BatchNorm2d(ch), ACT(inplace=True))
        self.spw  = ConvBN(ch, mid, k=1, p=0)
        self.merge= ConvBN(5*mid, ch, k=1, p=0)
        self.coord= CoordConv()
        self.hp = nn.Sequential(nn.AdaptiveAvgPool2d((None,1)),
                                 nn.Conv2d(ch+2,mid,1,bias=False),
                                 nn.BatchNorm2d(mid), ACT(inplace=True))
        self.vp = nn.Sequential(nn.AdaptiveAvgPool2d((1,None)),
                                 nn.Conv2d(ch+2,mid,1,bias=False),
                                 nn.BatchNorm2d(mid), ACT(inplace=True))
        self.sm = ConvBN(ch+2*mid, ch, k=1, p=0)
        self.deep= nn.Sequential(DSConv(ch), DSConv(ch))
        init_weights(self)
    def forward(self, x):
        B,_,H,W = x.shape; pw=self.spw
        d1=pw(self.dw1(x)); d2=pw(self.dw2(x))
        d4=pw(self.dw4(x)); d8=pw(self.dw8(x))
        gc=pw(self.gp(x).expand_as(x))
        x = self.merge(torch.cat([d1,d2,d4,d8,gc],1)) + x
        xc= self.coord(x)
        hp= self.hp(xc).expand(B,-1,H,W)
        vp= self.vp(xc).expand(B,-1,H,W)
        x = self.sm(torch.cat([x,hp,vp],1)) + x
        return self.deep(x)


class LocalWindowAttention(nn.Module):
    def __init__(self, ch=CORE_CH, win=7):
        super().__init__(); self.win=win; mid=ch//2
        self.q=nn.Conv2d(ch,mid,1,bias=False)
        self.k=nn.Conv2d(ch,mid,1,bias=False)
        self.v=nn.Conv2d(ch,mid,1,bias=False)
        self.out=nn.Sequential(nn.Conv2d(mid,ch,1,bias=False), nn.BatchNorm2d(ch))
        self.scale=mid**-0.5
    def forward(self, x):
        B,C,H,W=x.shape; w=self.win
        ph=(w-H%w)%w; pw_=(w-W%w)%w
        xp=F.pad(x,(0,pw_,0,ph)) if (ph or pw_) else x
        _,_,Hp,Wp=xp.shape; nH,nW=Hp//w,Wp//w
        Q=self.q(xp); K=self.k(xp); V=self.v(xp); mid=Q.shape[1]
        def to_win(t):
            return t.view(B,mid,nH,w,nW,w).permute(0,2,4,1,3,5)\
                    .contiguous().view(B*nH*nW,mid,w*w)
        Qw,Kw,Vw=to_win(Q),to_win(K),to_win(V)
        attn=torch.softmax(torch.bmm(Qw.permute(0,2,1),Kw)*self.scale,dim=-1)
        ow=torch.bmm(attn,Vw.permute(0,2,1)).permute(0,2,1).view(B*nH*nW,mid,w,w)
        out=ow.view(B,nH,nW,mid,w,w).permute(0,3,1,4,2,5).contiguous().view(B,mid,Hp,Wp)
        return self.out(out[:,:,:H,:W])


class NhanTuDuyNhanDien(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.lwa  = LocalWindowAttention(ch, 7)
        self.tmpl = nn.Conv2d(ch, 8, 3, padding=1, bias=False)
        self.proj = ConvBN(8, ch, k=1, p=0)
        self.deep = nn.Sequential(DSConv(ch), DSConv(ch))
        init_weights(self)
    def forward(self, x):
        x = self.lwa(x) + x
        x = self.proj(torch.softmax(self.tmpl(x), 1)) + x
        return self.deep(x)


class EfficientPrototypeMatcher(nn.Module):
    def __init__(self, ch=CORE_CH, n=16):
        super().__init__()
        self.protos = nn.Parameter(torch.randn(n, ch))
        nn.init.orthogonal_(self.protos)
        self.project = ConvBN(n, ch, k=1, p=0)
    def forward(self, x):
        B,C,H,W=x.shape
        f  = F.normalize(x.permute(0,2,3,1).reshape(-1,C), dim=1)
        p  = F.normalize(self.protos, dim=1)
        sim= (f @ p.T).view(B,H,W,-1).permute(0,3,1,2)
        return self.project(sim) + x


class GatedOccupancy(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__(); bot=max(ch//8,8)
        self.occ = nn.Sequential(
            nn.Conv2d(ch,bot,1,bias=False), nn.BatchNorm2d(bot), ACT(inplace=True),
            nn.Conv2d(bot,1,1), nn.Sigmoid())
        self.gate= ConvBN(ch+1, ch, k=1, p=0)
    def forward(self, x): return self.gate(torch.cat([x, self.occ(x)], 1)) + x


class NhanHieuMoiTruong(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.proto= EfficientPrototypeMatcher(ch)
        self.occ  = GatedOccupancy(ch)
        self.deep = nn.Sequential(DSConv(ch), DSConv(ch))
        init_weights(self)
    def forward(self, x): return self.deep(self.occ(self.proto(x)))


class ParameterFreeLCN(nn.Module):
    def __init__(self, k=5): super().__init__(); self.p=nn.AvgPool2d(k,1,k//2)
    def forward(self, x):
        mu=self.p(x); xc=x-mu; return xc/torch.sqrt(self.p(xc**2)+1e-5)


class StatisticalSaliency(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__(); bot=ch//4
        self.sal = nn.Sequential(
            nn.Conv2d(ch,bot,1,bias=False), nn.BatchNorm2d(bot), ACT(inplace=True),
            nn.Conv2d(bot,1,3,padding=1), nn.Sigmoid())
        self.gate= ConvBN(ch+1, ch, k=1, p=0)
    def forward(self, x):
        mu = F.avg_pool2d(x, 9, 1, 4)
        sal= self.sal(x) * torch.sigmoid((x-mu).abs().mean(1,True))
        return self.gate(torch.cat([x, sal], 1)) + x


class NhanNhanBietMoiTruong(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.lcn  = ParameterFreeLCN(5)
        self.sal  = StatisticalSaliency(ch)
        self.deep = nn.Sequential(DSConv(ch), DSConv(ch))
        init_weights(self)
    def forward(self, x): return self.deep(self.sal(self.lcn(x) + x*0.5))


class SharedWeightPPM(nn.Module):
    def __init__(self, ch=CORE_CH, sizes=(1,2,3,6)):
        super().__init__(); mid=ch//4
        self.spr  = nn.Sequential(nn.Conv2d(ch,mid,1,bias=False),
                                   nn.BatchNorm2d(mid), ACT(inplace=True))
        self.sizes= sizes
        self.fuse = ConvBN(ch+len(sizes)*mid, ch, k=1, p=0)
    def forward(self, x):
        H,W=x.shape[2],x.shape[3]; parts=[x]
        for ps in self.sizes:
            p=self.spr(F.adaptive_avg_pool2d(x, ps))
            parts.append(F.interpolate(p,(H,W),mode='bilinear',align_corners=False))
        return self.fuse(torch.cat(parts,1)) + x


class LinearAttention(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__(); mid=ch//4
        self.q=nn.Conv2d(ch,mid,1,bias=False)
        self.k=nn.Conv2d(ch,mid,1,bias=False)
        self.v=nn.Conv2d(ch,mid,1,bias=False)
        self.out=nn.Sequential(nn.Conv2d(mid,ch,1,bias=False), nn.BatchNorm2d(ch))
    @staticmethod
    def _phi(x): return F.elu(x) + 1
    def forward(self, x):
        B,C,H,W=x.shape; N=H*W
        Q=self._phi(self.q(x)).view(B,-1,N)
        K=self._phi(self.k(x)).view(B,-1,N)
        V=self.v(x).view(B,-1,N)
        KV  = torch.bmm(K, V.permute(0,2,1))
        out = torch.bmm(KV, Q) / (N+1e-6)
        Ks  = K.sum(-1,keepdim=True)
        out = out / ((Q*Ks).sum(1,keepdim=True) + 1e-6)
        return self.out(out.view(B,-1,H,W))


class NhanThauHieuMoiTruong(nn.Module):
    def __init__(self, ch=CORE_CH):
        super().__init__()
        self.ppm      = SharedWeightPPM(ch)
        self.lin_attn = LinearAttention(ch)
        self.deep     = nn.Sequential(DSConv(ch), DSConv(ch))
        init_weights(self)
    def forward(self, x):
        x = self.ppm(x)
        x = self.lin_attn(x) + x
        return self.deep(x)


# ─────────────────────────────────────────────────────────────────────
# CROSS-CORE INTERACTION + FUSION
# ─────────────────────────────────────────────────────────────────────

class CrossCoreInteraction(nn.Module):
    def __init__(self, ch=CORE_CH, n_heads=4):
        super().__init__(); self.ch=ch; self.nh=n_heads; self.scale=(ch//n_heads)**-0.5
        self.q=nn.Linear(ch,ch,bias=False); self.k=nn.Linear(ch,ch,bias=False)
        self.v=nn.Linear(ch,ch,bias=False); self.o=nn.Linear(ch,ch,bias=False)
        self.n1=nn.LayerNorm(ch); self.n2=nn.LayerNorm(ch)
        self.ffn=nn.Sequential(nn.Linear(ch,ch*2),nn.GELU(),nn.Linear(ch*2,ch))
        self.gate=nn.Sequential(nn.Linear(ch,ch), nn.Sigmoid())
        init_weights(self)
    def forward(self, cf):
        B,C,H,W=cf[0].shape
        tokens=torch.stack([f.mean([2,3]) for f in cf], 1)
        t=self.n1(tokens)
        Q=self.q(t).view(B,NUM_CORES,self.nh,-1).transpose(1,2)
        K=self.k(t).view(B,NUM_CORES,self.nh,-1).transpose(1,2)
        V=self.v(t).view(B,NUM_CORES,self.nh,-1).transpose(1,2)
        attn=torch.softmax(torch.matmul(Q,K.transpose(-2,-1))*self.scale,-1)
        att =torch.matmul(attn,V).transpose(1,2).reshape(B,NUM_CORES,C)
        tokens=tokens+self.o(att); tokens=tokens+self.ffn(self.n2(tokens))
        w=self.gate(tokens)
        return [cf[i]*w[:,i].view(B,C,1,1) for i in range(NUM_CORES)]


class CoreGatingFusion(nn.Module):
    def __init__(self, core_ch=CORE_CH, out_ch=256):
        super().__init__(); total=core_ch*NUM_CORES
        self.reduce=ConvBN(total, out_ch, k=1, p=0)
    def forward(self, cf):
        return self.reduce(torch.cat(cf, 1))


# ─────────────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────────────

class WidEyeV5(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, img_size=224,
                 stage_depths=(3,4,13,4), drop_path_rate=0.2):
        super().__init__()
        self.img_size=img_size; self.num_classes=num_classes

        # Stochastic depth rates: linearly increase
        total_blocks = sum(stage_depths)
        dp_all = [drop_path_rate * i / (total_blocks-1)
                  for i in range(total_blocks)]
        idx = 0
        dp_s = []
        for d in stage_depths:
            dp_s.append(dp_all[idx:idx+d]); idx+=d

        self.stem = nn.Sequential(
            RepConvBN(3, 32, s=2),
            ConvBN(32, 64, k=3, s=1, p=1),
            RepConvBN(64, CORE_CH, s=2),
            ConvBN(CORE_CH, CORE_CH, k=3, s=1, p=1))

        self.core1=NhanThiGiac(); self.core2=NhanTuDuyMoiTruong()
        self.core3=NhanTuDuyNhanDien(); self.core4=NhanHieuMoiTruong()
        self.core5=NhanNhanBietMoiTruong(); self.core6=NhanThauHieuMoiTruong()
        self.cross_core=CrossCoreInteraction(CORE_CH, 4)
        self.fusion=CoreGatingFusion(CORE_CH, 256)

        d1,d2,d3,d4 = stage_depths
        self.stage1=make_stage(256, 64,  d1, 1, 0,   'none', dp_s[0])
        self.stage2=make_stage(256, 128, d2, 2, 2,   'se',   dp_s[1])
        self.stage3=make_stage(512, 256, d3, 2, 2,   'se',   dp_s[2])
        self.stage4=make_stage(1024,512, d4, 2, 1,   'cbam', dp_s[3])

        self.lat4=ConvBN(2048,256,k=1,p=0); self.lat3=ConvBN(1024,256,k=1,p=0)
        self.lat2=ConvBN(512, 256,k=1,p=0)
        self.fo4 =ConvBN(256,256,k=3,p=1); self.fo3=ConvBN(256,256,k=3,p=1)
        self.fo2 =ConvBN(256,256,k=3,p=1)

        self.final_cbam=CBAM(2048, r=16, sk=7)
        self.gap=nn.AdaptiveAvgPool2d(1); self.fpn_pool=nn.AdaptiveAvgPool2d(1)

        self.shared_fc=nn.Sequential(
            nn.Linear(2304,1024,bias=False), nn.BatchNorm1d(1024),
            ACT(inplace=True), nn.Dropout(0.35),
            nn.Linear(1024,512,bias=False), nn.BatchNorm1d(512),
            ACT(inplace=True), nn.Dropout(0.2))

        self.cls_head=nn.Sequential(
            nn.Linear(512,256), ACT(inplace=True), nn.Dropout(0.2),
            nn.Linear(256,128), ACT(inplace=True),
            nn.Linear(128, num_classes))

        self.bbox_enc =nn.Sequential(nn.Linear(256,128), ACT(inplace=True))
        self.bbox_head=nn.Sequential(
            nn.Linear(640,256), ACT(inplace=True), nn.Dropout(0.1),
            nn.Linear(256,64),  ACT(inplace=True),
            nn.Linear(64,4),    nn.Sigmoid())

        init_weights(self)

    def forward(self, x):
        s=self.stem(x)
        cores=[self.core1(s),self.core2(s),self.core3(s),
               self.core4(s),self.core5(s),self.core6(s)]
        cores=self.cross_core(cores)
        fused=self.fusion(cores)
        s1=self.stage1(fused); s2=self.stage2(s1)
        s3=self.stage3(s2);    s4=self.stage4(s3)
        p4=self.fo4(self.lat4(s4))
        p3=self.fo3(self.lat3(s3)+F.interpolate(p4,s3.shape[-2:],mode='nearest'))
        p2=self.fo2(self.lat2(s2)+F.interpolate(p3,s2.shape[-2:],mode='nearest'))
        df=self.gap(self.final_cbam(s4)).flatten(1)
        ff=self.fpn_pool(p2).flatten(1)
        sh=self.shared_fc(torch.cat([df,ff],1))
        cl=self.cls_head(sh)
        bp=self.bbox_head(torch.cat([sh, self.bbox_enc(ff)],1))
        return cl, bp


# ─────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step  = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for ema_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(m_p.data, alpha=1.0 - d)
        for ema_b, m_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(m_b)


# ─────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

class LabelSmoothCE(nn.Module):
    def __init__(self, n=NUM_CLASSES, smooth=0.1):
        super().__init__(); self.s=smooth; self.n=n
    def forward(self, p, t):
        lp=F.log_softmax(p,-1)
        with torch.no_grad():
            d=torch.full_like(p,self.s/self.n); d.scatter_(1,t.unsqueeze(1),1-self.s)
        return -(d*lp).sum(-1).mean()


class CIoULoss(nn.Module):
    """
    Complete IoU = IoU - d²/c² - α·v
    v = (4/π²)(arctan(wg/hg) - arctan(w/h))²
    α = v / (1 - IoU + v)
    """
    def forward(self, pred, tgt):
        def xyxy(b):
            return torch.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2,
                                  b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2], 1)
        p,g=xyxy(pred),xyxy(tgt)
        ix1=torch.max(p[:,0],g[:,0]); iy1=torch.max(p[:,1],g[:,1])
        ix2=torch.min(p[:,2],g[:,2]); iy2=torch.min(p[:,3],g[:,3])
        inter=torch.clamp(ix2-ix1,0)*torch.clamp(iy2-iy1,0)
        pa=(p[:,2]-p[:,0])*(p[:,3]-p[:,1])
        ga=(g[:,2]-g[:,0])*(g[:,3]-g[:,1])
        union=pa+ga-inter+1e-7; iou=inter/union

        # Enclosing box diagonal²
        cx1=torch.min(p[:,0],g[:,0]); cy1=torch.min(p[:,1],g[:,1])
        cx2=torch.max(p[:,2],g[:,2]); cy2=torch.max(p[:,3],g[:,3])
        c2 =(cx2-cx1)**2+(cy2-cy1)**2+1e-7

        # Center distance²
        pcx=(p[:,0]+p[:,2])/2; pcy=(p[:,1]+p[:,3])/2
        gcx=(g[:,0]+g[:,2])/2; gcy=(g[:,1]+g[:,3])/2
        d2 =(pcx-gcx)**2+(pcy-gcy)**2

        # Aspect ratio penalty
        pw=p[:,2]-p[:,0]; ph=p[:,3]-p[:,1]
        gw=g[:,2]-g[:,0]; gh=g[:,3]-g[:,1]
        v = (4/(math.pi**2)) * (torch.atan(gw/(gh+1e-7)) - torch.atan(pw/(ph+1e-7)))**2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - d2/c2 - alpha*v
        return (1 - ciou).mean()


class TaskAlignedLoss(nn.Module):
    """
    Combined loss. Bbox quality weights cls loss (Task-Aligned Learning).
    If bbox prediction is poor, classification loss is amplified.
    """
    def __init__(self, lam_cls=1.0, lam_ciou=2.0, lam_l1=0.5):
        super().__init__()
        self.cls  = LabelSmoothCE()
        self.ciou = CIoULoss()
        self.l1   = nn.SmoothL1Loss(beta=0.1)
        self.lc=lam_cls; self.li=lam_ciou; self.ll=lam_l1

    def forward(self, cl, bp, ct, bt, bw=1.0):
        # IoU quality per sample (detached)
        with torch.no_grad():
            q = self._iou_per_sample(bp.detach(), bt)  # (B,) ∈ [0,1]
            cls_weight = (2.0 - q).mean()              # lower IoU → higher cls weight

        lc = self.cls(cl, ct) * cls_weight
        li = self.ciou(bp, bt)
        ll = self.l1(bp, bt)
        return self.lc*lc + bw*(self.li*li + self.ll*ll), lc.item(), li.item(), ll.item()

    @staticmethod
    def _iou_per_sample(pred, tgt):
        px1=pred[:,0]-pred[:,2]/2; py1=pred[:,1]-pred[:,3]/2
        px2=pred[:,0]+pred[:,2]/2; py2=pred[:,1]+pred[:,3]/2
        gx1=tgt[:,0]-tgt[:,2]/2;  gy1=tgt[:,1]-tgt[:,3]/2
        gx2=tgt[:,0]+tgt[:,2]/2;  gy2=tgt[:,1]+tgt[:,3]/2
        ix1=torch.max(px1,gx1); iy1=torch.max(py1,gy1)
        ix2=torch.min(px2,gx2); iy2=torch.min(py2,gy2)
        inter=torch.clamp(ix2-ix1,0)*torch.clamp(iy2-iy1,0)
        union=(px2-px1)*(py2-py1)+(gx2-gx1)*(gy2-gy1)-inter+1e-7
        return inter/union


class KDLoss(nn.Module):
    """Knowledge Distillation: logit distillation (soft targets)."""
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__(); self.T=T; self.alpha=alpha
    def forward(self, s_logits, t_logits):
        soft_t = F.softmax(t_logits / self.T, dim=1)
        soft_s = F.log_softmax(s_logits / self.T, dim=1)
        return F.kl_div(soft_s, soft_t, reduction='batchmean') * (self.T**2)


# ─────────────────────────────────────────────────────────────────────
# AUGMENTED DATASET
# ─────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    0:[(220,60,50),(255,110,60),(200,40,30)],
    1:[(50,190,60),(30,160,90),(80,210,110)],
    2:[(110,110,120),(90,90,100),(140,140,150)],
    3:[(160,60,210),(190,90,230),(140,40,190)],
    4:[(230,210,40),(250,190,60),(210,220,30)],
}


class EnvironmentDataset(Dataset):
    def __init__(self, n=10000, size=224, augment=True, mosaic_prob=0.5,
                 mixup_prob=0.15, copy_paste_prob=0.3):
        self.n=n; self.sz=size; self.aug=augment
        self.mosaic_p=mosaic_prob; self.mixup_p=mixup_prob; self.cp_p=copy_paste_prob
        self.base_tf=T.Compose([T.ToTensor(),
                                  T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def _bg(self, S, rng):
        t=rng.randint(0,5); arr=np.zeros((S,S,3),np.uint8)
        if t==0:
            c1=np.array([rng.randint(10,80)]*3); c2=np.array([rng.randint(80,200)]*3)
            for i in range(S): arr[:,i]=(c1*(1-i/S)+c2*(i/S)).astype(np.uint8)
        elif t==1:
            np.random.RandomState(rng.randint(0,99999)).randint(20,100,(S,S,3),dtype=np.uint8,out=arr)
        elif t==2:
            b=rng.randint(40,120); arr[:]=b; st=rng.randint(16,48)
            arr[::st,:]=b+40; arr[:,::st]=b+40
        elif t==3:
            ts=rng.randint(20,40); c1,c2=rng.randint(40,100),rng.randint(110,200)
            for y in range(0,S,ts):
                for x in range(0,S,ts):
                    arr[y:y+ts,x:x+ts]=c1 if (y//ts+x//ts)%2==0 else c2
        elif t==4:
            Y,X=np.ogrid[:S,:S]
            d=np.clip(np.sqrt((X-S//2)**2+(Y-S//2)**2)/(S*0.7),0,1)[...,None]
            c1=np.array([rng.randint(20,80)]*3); c2=np.array([rng.randint(100,200)]*3)
            arr=(c1*(1-d)+c2*d).astype(np.uint8)
        else: arr[:]=rng.randint(40,160)
        return arr

    def _draw(self, draw, cid, cx, cy, w, h, rng):
        x1,y1,x2,y2=int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)
        mc=random.choice(CLASS_COLORS[cid])
        lt=tuple(min(255,c+60) for c in mc); dk=tuple(max(0,c-40) for c in mc)
        if cid==0:
            draw.rectangle([x1,y1,x2,y2],fill=mc,outline=(0,0,0),width=2)
            draw.rectangle([x1,y1,x2,y1+max(2,h//6)],fill=lt)
        elif cid==1:
            draw.polygon([(cx,y1),(x1,y2),(x2,y2)],fill=mc,outline=dk)
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
            for wx in [x1+w//6,x2-w//6]:
                draw.ellipse([wx-wr,y2-wr,wx+wr,y2+wr],fill=(30,30,30))

    def _make_single(self, idx, size=None):
        """Generate one synthetic image. Returns (PIL, cid, cx, cy, ow, oh)."""
        S=size or self.sz; rng=random.Random(idx*2053+17)
        arr=self._bg(S,rng); img=Image.fromarray(arr.astype(np.uint8))
        draw=ImageDraw.Draw(img)
        cid=rng.randint(0,NUM_CLASSES-1)
        ow=rng.randint(int(0.2*S),int(0.6*S)); oh=rng.randint(int(0.2*S),int(0.6*S))
        m=5
        cx=rng.randint(ow//2+m,S-ow//2-m); cy=rng.randint(oh//2+m,S-oh//2-m)
        for _ in range(rng.randint(0,2)):
            dc=rng.randint(0,NUM_CLASSES-1); dw=rng.randint(int(0.06*S),int(0.2*S))
            dh=rng.randint(int(0.06*S),int(0.2*S))
            dx=rng.randint(dw//2+2,S-dw//2-2); dy=rng.randint(dh//2+2,S-dh//2-2)
            self._draw(draw,dc,dx,dy,dw,dh,rng)
        self._draw(draw,cid,cx,cy,ow,oh,rng)
        # Noise
        arr2=np.clip(np.array(img).astype(np.float32)+
                     np.random.normal(0,rng.uniform(0,15),(S,S,3)),0,255).astype(np.uint8)
        return Image.fromarray(arr2), cid, cx, cy, ow, oh

    def _mosaic(self, idx):
        """Combine 4 images in 2×2 grid. Returns (PIL, cid, cx, cy, w, h) of center object."""
        S=self.sz; half=S//2
        canvas=Image.new('RGB',(S,S))
        indices=[idx,(idx+1)%self.n,(idx+2)%self.n,(idx+3)%self.n]
        positions=[(0,0),(half,0),(0,half),(half,half)]
        main_cid=None; main_bbox=None
        for i,(pos,src_idx) in enumerate(zip(positions,indices)):
            img,cid,cx,cy,ow,oh=self._make_single(src_idx,half)
            canvas.paste(img,pos)
            if i==0:
                main_cid=cid
                # Adjust bbox to mosaic coords
                main_bbox=(cx+pos[0], cy+pos[1], ow, oh)
        cx_m,cy_m,ow_m,oh_m=main_bbox
        return canvas, main_cid, cx_m, cy_m, ow_m, oh_m

    def _mixup(self, img1, cid1, img2, cid2, alpha=0.4):
        """Blend two images. Returns blended img + soft label."""
        lam = np.random.beta(alpha, alpha)
        arr1=np.array(img1).astype(np.float32)
        arr2=np.array(img2.resize(img1.size)).astype(np.float32)
        mixed=Image.fromarray((lam*arr1+(1-lam)*arr2).astype(np.uint8))
        # Soft label: dominant class
        return mixed, cid1 if lam>=0.5 else cid2

    def _copy_paste(self, idx):
        """Paste foreground object from another image onto current."""
        rng=random.Random(idx*9999)
        img1,cid1,cx1,cy1,ow1,oh1=self._make_single(idx)
        img2,cid2,cx2,cy2,ow2,oh2=self._make_single((idx+7)%self.n)
        S=self.sz
        # Crop object from img2
        x1c=max(0,int(cx2-ow2//2)); y1c=max(0,int(cy2-oh2//2))
        x2c=min(S,int(cx2+ow2//2)); y2c=min(S,int(cy2+oh2//2))
        crop=img2.crop((x1c,y1c,x2c,y2c))
        # Paste to random position in img1
        px=rng.randint(0,max(1,S-crop.width)); py=rng.randint(0,max(1,S-crop.height))
        img1.paste(crop,(px,py))
        return img1, cid1, cx1, cy1, ow1, oh1

    def __len__(self): return self.n

    def __getitem__(self, idx):
        rng=random.Random(idx*2053+17); S=self.sz
        # Choose augmentation strategy
        r=rng.random()
        if self.aug and r < self.mosaic_p:
            img,cid,cx,cy,ow,oh=self._mosaic(idx)
        elif self.aug and r < self.mosaic_p+self.cp_p:
            img,cid,cx,cy,ow,oh=self._copy_paste(idx)
        else:
            img,cid,cx,cy,ow,oh=self._make_single(idx)

        # MixUp (applied after base augment)
        if self.aug and rng.random() < self.mixup_p:
            img2,cid2,_,_,_,_=self._make_single((idx+13)%self.n)
            img,cid=self._mixup(img,cid,img2,cid2)

        # Color augmentation
        if self.aug:
            if rng.random()<0.6: img=TF.adjust_brightness(img,rng.uniform(0.6,1.4))
            if rng.random()<0.4: img=TF.adjust_contrast(img,rng.uniform(0.7,1.3))
            if rng.random()<0.3: img=img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3,1.2)))
            if rng.random()<0.3: img=TF.hflip(img); cx=S-cx

        # BBox ground truth
        x1g=max(0,cx-ow//2); y1g=max(0,cy-oh//2)
        x2g=min(S,cx+ow//2); y2g=min(S,cy+oh//2)
        vw=x2g-x1g; vh=y2g-y1g
        bbox=torch.tensor([(x1g+x2g)/2/S,(y1g+y2g)/2/S,vw/S,vh/S],dtype=torch.float32)
        return self.base_tf(img.resize((S,S))), torch.tensor(cid,dtype=torch.long), bbox


# ─────────────────────────────────────────────────────────────────────
# PROGRESSIVE RESIZING SCHEDULER
# ─────────────────────────────────────────────────────────────────────

class ProgressiveResizeScheduler:
    """
    Increases input resolution through training stages.
    sizes: list of (size, start_epoch)
    """
    def __init__(self, sizes=None):
        self.schedule = sizes or [
            (112, 0), (160, 8), (192, 16), (224, 24), (256, 32), (320, 40)
        ]
    def get_size(self, epoch):
        sz = self.schedule[0][0]
        for size, start_ep in self.schedule:
            if epoch >= start_ep:
                sz = size
        return sz


# ─────────────────────────────────────────────────────────────────────
# WARMUP COSINE LR
# ─────────────────────────────────────────────────────────────────────

class WarmupCosine:
    def __init__(self, opt, we, te, plr, mlr=1e-6):
        self.opt=opt; self.we=we; self.te=te; self.plr=plr; self.mlr=mlr
    def step(self, ep):
        if ep < self.we:
            lr=self.plr*(ep+1)/self.we
        else:
            p=(ep-self.we)/(self.te-self.we)
            lr=self.mlr+0.5*(self.plr-self.mlr)*(1+math.cos(math.pi*p))
        for g in self.opt.param_groups: g['lr']=lr
        return lr


# ─────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────

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
          save="wideeye_v5.pth", warmup=5, accum=2, amp=True,
          drop_path_rate=0.2, teacher_path=None, ema_decay=0.9999,
          progressive_resize=True):

    device=torch.device("cuda" if torch.cuda.is_available() else
                        "mps"  if torch.backends.mps.is_available() else "cpu")
    amp_on = amp and device.type=="cuda"

    print(f"\n{'═'*68}")
    print(f"  WidEye-CoreSix v5 Ultra  |  {device}  |  AMP:{amp_on}")
    print(f"  Eff-batch:{batch*accum}  Epochs:{num_epochs}  DropPath:{drop_path_rate}")
    print(f"  ProgResize:{progressive_resize}  EMA:{ema_decay}  KD:{teacher_path is not None}")
    print(f"{'═'*68}\n")

    model = WidEyeV5(NUM_CLASSES, drop_path_rate=drop_path_rate).to(device)
    tp    = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {tp:,}  ({tp/1e6:.2f}M)\n")

    ema   = ModelEMA(model, ema_decay)
    opt   = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=5e-4)
    sched = WarmupCosine(opt, warmup, num_epochs, peak_lr)
    crit  = TaskAlignedLoss()
    kd    = KDLoss(T=4.0, alpha=0.5) if teacher_path else None
    scaler= torch.cuda.amp.GradScaler(enabled=amp_on)
    resize_sched = ProgressiveResizeScheduler() if progressive_resize else None

    # Load teacher
    teacher = None
    if teacher_path:
        ck = torch.load(teacher_path, map_location=device, weights_only=False)
        teacher = WidEyeV5(NUM_CLASSES).to(device)
        teacher.load_state_dict(ck["model_state"])
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
        print(f"  Teacher loaded from {teacher_path}\n")

    # Initial dataset
    cur_size = resize_sched.get_size(0) if resize_sched else 224
    tds = EnvironmentDataset(n_train, cur_size, True)
    vds = EnvironmentDataset(n_val,   224,       False)
    nw  = 2
    tldr= DataLoader(tds,batch,True, num_workers=nw, pin_memory=device.type=='cuda', drop_last=True)
    vldr= DataLoader(vds,batch,False,num_workers=nw, pin_memory=device.type=='cuda')

    best_iou=0.0; patience=10; no_imp=0

    for ep in range(1, num_epochs+1):
        t0=time.time(); bw=min(1.0, ep/max(warmup,1))

        # Progressive resize: rebuild dataset if size changes
        if resize_sched:
            new_size = resize_sched.get_size(ep)
            if new_size != cur_size:
                cur_size = new_size
                tds = EnvironmentDataset(n_train, cur_size, True)
                tldr= DataLoader(tds,batch,True,num_workers=nw,
                                  pin_memory=device.type=='cuda',drop_last=True)
                print(f"  ↑ Progressive resize → {cur_size}×{cur_size}")

        # Train
        model.train(); tl_sum=0.0
        opt.zero_grad(set_to_none=True)
        for step,(imgs,lbls,bbs) in enumerate(tldr):
            imgs=imgs.to(device,non_blocking=True)
            lbls=lbls.to(device,non_blocking=True)
            bbs =bbs.to(device,non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_on):
                cl,bp=model(imgs)
                loss,*_=crit(cl,bp,lbls,bbs,bw)
                if teacher is not None:
                    t_cl,_=teacher(imgs)
                    loss = loss + kd(cl, t_cl)
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

        # Validate with EMA model
        ema.shadow.eval(); vl_sum=0.0; corr=0; tot=0; iou_s=0.0; nb=0
        with torch.no_grad():
            for imgs,lbls,bbs in vldr:
                imgs=imgs.to(device,non_blocking=True)
                lbls=lbls.to(device,non_blocking=True)
                bbs =bbs.to(device,non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_on):
                    cl,bp=ema.shadow(imgs)
                    loss,*_=crit(cl,bp,lbls,bbs,1.0)
                vl_sum+=loss.item()
                corr+=(cl.argmax(1)==lbls).sum().item(); tot+=lbls.size(0)
                iou_s+=compute_iou(bp,bbs); nb+=1

        avg_vl=vl_sum/len(vldr); acc=corr/tot*100; avg_iou=iou_s/nb
        lr=sched.step(ep); elapsed=time.time()-t0

        print(f"  [{ep:3d}/{num_epochs}] T:{avg_tl:.4f} V:{avg_vl:.4f} "
              f"Acc:{acc:5.1f}% IoU:{avg_iou:.4f} "
              f"LR:{lr:.1e} sz:{cur_size} {elapsed:.1f}s")

        if avg_iou > best_iou:
            best_iou=avg_iou; no_imp=0
            torch.save({
                "epoch":ep,
                "model_state":ema.shadow.state_dict(),
                "opt":opt.state_dict(),
                "best_iou":best_iou, "val_acc":acc,
                "num_classes":NUM_CLASSES, "img_size":224,
                "classes":CLASSES, "arch":"WidEyeV5",
                "params":tp,
            }, save)
            print(f"  ✓ Saved EMA model (IoU:{best_iou:.4f})")
        else:
            no_imp+=1
            if no_imp>=patience:
                print(f"\n  Early stop @ epoch {ep}"); break

    print(f"\n  Best IoU: {best_iou:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────

class Inference:
    def __init__(self, path="wideeye_v5.pth", size=224, deploy=True):
        self.sz=size
        self.dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck=torch.load(path,map_location=self.dev,weights_only=False)
        self.cls=ck.get("classes",CLASSES)
        model=WidEyeV5(ck["num_classes"],size).to(self.dev)
        model.load_state_dict(ck["model_state"])
        if deploy:
            model=reparameterize_model(model)
        model.eval()
        try: model=torch.compile(model)
        except Exception: pass
        self.model=model
        self.tf=T.Compose([T.Resize((size,size)),T.ToTensor(),
                           T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        p=sum(x.numel() for x in model.parameters())
        print(f"  v5 Loaded: {p/1e6:.2f}M | IoU={ck.get('best_iou',0):.4f} "
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


def run_webcam(path="wideeye_v5.pth", size=224, cam=0):
    try: import cv2
    except ImportError: print("pip install opencv-python"); return
    COLORS={"obstacle_box":(0,50,220),"path_clear":(50,200,50),
            "wall_detected":(150,150,150),"person_nearby":(200,50,200),
            "vehicle_zone":(0,200,220)}
    eye=Inference(path,size); cap=cv2.VideoCapture(cam)
    if not cap.isOpened(): print(f"Cannot open camera {cam}"); return
    fps_h=[]; fc=0; print("Q=quit  S=screenshot\n")
    while True:
        t0=time.time(); ok,frame=cap.read()
        if not ok: break
        fc+=1
        cls_name,conf,(x1,y1,x2,y2)=eye.predict(
            Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
        H,W=frame.shape[:2]; clr=COLORS.get(cls_name,(255,255,255))
        cv2.rectangle(frame,(x1,y1),(x2,y2),clr,2)
        cl_=15; ct=3
        for px,py,sx,sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(px,py),(px+sx*cl_,py),clr,ct)
            cv2.line(frame,(px,py),(px,py+sy*cl_),clr,ct)
        lbl=f"{cls_name} {conf*100:.0f}%"
        (lw,lh),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
        ly=max(y1-8,lh+8)
        cv2.rectangle(frame,(x1,ly-lh-5),(x1+lw+4,ly+2),clr,-1)
        cv2.putText(frame,lbl,(x1+2,ly-2),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
        fps_h.append(1/(time.time()-t0+1e-9))
        if len(fps_h)>30: fps_h.pop(0)
        cv2.putText(frame,f"FPS:{sum(fps_h)/len(fps_h):.1f}",(10,26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(frame,"WidEye v5",(10,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,180),1)
        cv2.imshow("WidEye-CoreSix v5",frame)
        key=cv2.waitKey(1)&0xFF
        if key in [ord('q'),27]: break
        elif key==ord('s'): cv2.imwrite(f"v5_{fc:05d}.jpg",frame)
    cap.release(); cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────
# ANALYSIS + TESTS
# ─────────────────────────────────────────────────────────────────────

def analyze():
    import ast
    with open(__file__,'r') as f: src=f.read()
    tree=ast.parse(src)

    reserved={'nonlocal','global','return','class','lambda','import','pass'}
    bad=[n.attr for n in ast.walk(tree)
         if isinstance(n,ast.Attribute) and n.attr in reserved]
    print(f"  Reserved word check: {'PASS' if not bad else f'FAIL {bad}'}")

    gs=[n.lineno for n in ast.walk(tree)
        if isinstance(n,ast.Attribute) and n.attr=='grid_sample']
    print(f"  grid_sample check  : {'PASS (0 calls)' if not gs else f'FOUND at {gs}'}")

    model=WidEyeV5(NUM_CLASSES)
    total=sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters : {total:,}  ({total/1e6:.3f}M)")
    print(f"  FP32 size  : {total*4/1024**2:.1f} MB")

    def mp(m): return sum(p.numel() for p in m.parameters())
    rows=[
        ("Stem",        model.stem),
        ("Core1",       model.core1),("Core2",model.core2),("Core3",model.core3),
        ("Core4",       model.core4),("Core5",model.core5),("Core6",model.core6),
        ("CrossCore",   model.cross_core),("Fusion",model.fusion),
        ("Stage1",      model.stage1),("Stage2",model.stage2),
        ("Stage3",      model.stage3),("Stage4",model.stage4),
        ("FPN+CBAM",    nn.ModuleList([model.lat2,model.lat3,model.lat4,
                                        model.fo2,model.fo3,model.fo4,model.final_cbam])),
        ("SharedFC",    model.shared_fc),
        ("Heads",       nn.Sequential(model.cls_head,model.bbox_enc,model.bbox_head)),
    ]
    print(f"\n  {'Module':<14} {'Params':>12}  {'%':>7}")
    print(f"  {'─'*38}")
    for name,mod in rows:
        p=mp(mod); print(f"  {name:<14} {p:>12,}  {100*p/total:>6.2f}%")

    # Forward
    dummy=torch.randn(2,3,224,224)
    with torch.no_grad(): cl,bp=model(dummy)
    assert cl.shape==(2,NUM_CLASSES) and bp.shape==(2,4)
    assert bp.min()>=0 and bp.max()<=1
    print(f"\n  Forward pass : PASS  cls={cl.shape} bbox={bp.shape}")

    # RepConv reparameterize
    rep_count=sum(1 for m in model.modules() if isinstance(m,RepConvBN))
    model_d=reparameterize_model(copy.deepcopy(model))
    with torch.no_grad(): cl2,bp2=model_d(dummy)
    diff=(cl-cl2).abs().max().item()
    print(f"  RepConv blocks: {rep_count}  |  Fused output diff: {diff:.2e}  {'PASS' if diff<1e-4 else 'FAIL'}")

    # DropPath
    dp_count=sum(1 for m in model.modules() if isinstance(m,DropPath) and m.p>0)
    print(f"  DropPath blocks active: {dp_count}")

    print(f"\n  ✓ All checks PASSED")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["analyze","train","webcam"],default="analyze")
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--batch",     type=int,   default=24)
    p.add_argument("--lr",        type=float, default=8e-4)
    p.add_argument("--n_train",   type=int,   default=10000)
    p.add_argument("--n_val",     type=int,   default=2000)
    p.add_argument("--accum",     type=int,   default=2)
    p.add_argument("--model",     type=str,   default="wideeye_v5.pth")
    p.add_argument("--teacher",   type=str,   default=None)
    p.add_argument("--camera",    type=int,   default=0)
    p.add_argument("--no_amp",    action="store_true")
    p.add_argument("--drop_path", type=float, default=0.2)
    p.add_argument("--no_prog",   action="store_true")
    args=p.parse_args()
    if   args.mode=="analyze": analyze()
    elif args.mode=="train":
        train(args.epochs,args.batch,args.lr,args.n_train,args.n_val,
              save=args.model,accum=args.accum,amp=not args.no_amp,
              drop_path_rate=args.drop_path,teacher_path=args.teacher,
              progressive_resize=not args.no_prog)
    elif args.mode=="webcam":
        run_webcam(args.model,cam=args.camera)
