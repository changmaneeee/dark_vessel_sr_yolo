'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat



class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = x.permute(0,3,1,2)
        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        self.dt_rank = math.ceil(d_model/16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner*2, bias = False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels = self.d_inner,
            kernel_size=4,
            padding=3,
            groups = self.d_inner,
            bias=True
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + d_state *2,
            bias = False
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A_init = repeat(torch.arange(1, d_state +1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A_init))

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()
    
    def _init_weights(self):
        dt_init_std = 2**-4
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
    def forward(self, x):
        xz = self.in_proj(x)
        x,z = xz.chunk(2, dim=-1)

        x=x.transpose(1,2)
        x = self.conv1d(x)[:, :, :x.shape[2]]
        x = x.transpose(1,2)

        x = F.silu(x)

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log)

        y = self._selective_scan(x, dt, A, B, C, self.D)

        y = y*F.silu(z)
        y = self.out_proj(y)
        return y

    def _selective_scan(self, u, dt, A, B, C, D):

        batch, seq_len, d_inner = u.shape

        d_state = A.shape[1]

        dA = torch.exp(torch.einsum('b l d, d n -> b l d n', dt, A))

        dB = torch.einsum('b l d, b l n -> b l d n', dt, B)

        h = torch.zeros(batch, d_inner, d_state, device=u.device)
        ys = []

        for i in range(seq_len):
            u_i = u[:, i, :] 
            h = dA[:, i] * h + dB[:,i]*u_i.unsqueeze(-1)
            y_i = torch.einsum('b d n, b n -> b d', h, C[:, i])
            y_i = y_i + D * u_i
            ys.append(y_i)

        return torch.stack(ys, dim=1)

class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.scan_forward_h = SelectiveScan(d_model, d_state, expand)  # →
        self.scan_backward_h = SelectiveScan(d_model, d_state, expand) # ←
        self.scan_forward_v = SelectiveScan(d_model, d_state, expand)  # ↓
        self.scan_backward_v = SelectiveScan(d_model, d_state, expand) # ↑
        
        self.out_proj = nn.Linear(d_model * 4, d_model)
    
    def forward(self, x):
        B,H,W,C = x.shape
        x_flat = rearrange(x, 'b h w c -> (b h) w c')
        y_h_f = self.scan_forward_h(x_flat)
        y_h_f = rearrange(y_h_f, '(b h) w c -> b h w c', b =B, h=H)

        x_flip = torch.flip(x_flat, dims=[1])
        y_h_b = self.scan_backward_h(x_flip)
        y_h_b = torch.flip(y_h_b, dims=[1])
        y_h_b = rearrange(y_h_b, '(b h) w c -> b h w c', b =B, h=H)

        x_v = rearrange(x, 'b h w c -> (b w) h c')
        y_v_f = self.scan_forward_v(x_v)
        y_v_f = rearrange(y_v_f, '(b w) h c -> b h w c', b=B, w=W)

        x_v_flip = torch.flip(x_v, dims=[1])
        y_v_b = self.scan_backward_v(x_v_flip)
        y_v_b = torch.flip(y_v_b, dims=[1])
        y_v_b = rearrange(y_v_b, '(b w) h c -> b h w c', b=B, w=W)

        y= torch.cat([y_h_f, y_h_b, y_v_f, y_v_b], dim=-1)
        y = self.out_proj(y)

        return y
    
class VSSBlock(nn.Module):
    def __init__(self, dim , d_state=16, expand=2, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state, expand)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop)

    def forward(self, x):

        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class MambaSR(nn.Module):
    def __init__(self,
                 in_channels: int =3,
                 out_channels: int =3,
                 dim =48,
                 n_blocks=6,
                 upscale =4):
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, dim, 3, 1, 1)

        self.body = nn.ModuleList([ 
            VSSBlock(dim = dim) for _ in range(n_blocks)
        ])

        self.conv_body = nn.Conv2d(dim, dim, 3, 1, 1)

        m_upsample = []
        if (upscale & (upscale -1)) == 0:
            for _ in range(int(math.log2(upscale))):
                m_upsample.append(nn.Conv2d(dim, dim *4, 3,1,1))
                m_upsample.append(nn.PixelShuffle(2))
                m_upsample.append(nn.GELU())
        
        self.upsample = nn.Sequential(*m_upsample)
        self.conv_last = nn.Conv2d(dim, out_channels, 3,1,1)

    def forward(self, x):

        feat_first = self.conv_first(x)
        x_in = feat_first.permute(0,2,3,1)

        for block in self.body:
            x_in = block(x_in)

        feat_body = x_in.permute(0,3,1,2)
        feat_body = self.conv_body(feat_body)

        feat = feat_body + feat_first
        out = self.upsample(feat)
        out = self.conv_last(out)

        return out
    
    def forward_features(self, x, layer_idx=-1):

        feat_first = self.conv_first(x)
        x_in = feat_first.permute(0,2,3,1)

        for i, block in enumerate(self.body):
            x_in = block(x_in)
        
        feat_body = x_in.permute(0,3,1,2)
        feat = feat_body + self.conv_body(feat_body)

        return feat

if __name__ == "__main__":
    print("🚀 MambaSR 모델 테스트 시작...")
    
    # 1. 모델 생성
    model = MambaSR(dim=48, n_blocks=4, upscale=4) # 가볍게 4블록만
    print(f"✅ 모델 생성 완료! (파라미터 수: {sum(p.numel() for p in model.parameters()):,})")
    
    # 2. 더미 데이터 입력
    x = torch.randn(1, 3, 64, 64) # 64x64 이미지
    print(f"📥 입력 크기: {x.shape}")
    
    # 3. 추론 테스트
    try:
        y = model(x)
        print(f"📤 출력 크기: {y.shape}") # 예상: (1, 3, 256, 256)
        
        if y.shape == (1, 3, 256, 256):
            print("🎉 테스트 성공! Mamba가 정상적으로 작동합니다.")
        else:
            print("⚠️ 출력 크기가 예상과 다릅니다.")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

'''


import torch
import torch.nn as nn
# 방금 우리가 수술한 파일에서 클래스를 가져옵니다.
from .mamba_archs.mambairv2light_arch import MambaIRv2Light 

class MambaIRDetector(nn.Module):
    """
    Arch 5-B용 Wrapper Class
    역할: MambaIRv2Light 모델을 생성하고, Arch 5-B에 맞는 입출력을 제공
    """
    def __init__(
        self, 
        upscale=4,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        **kwargs # Config에서 넘어오는 기타 잡동사니 인자 무시
    ):
        super().__init__()
        
        # MambaIRv2Light 모델 생성 (수술한 파일 사용)
        self.model = MambaIRv2Light(
            upscale=upscale,
            img_size=img_size,
            embed_dim=embed_dim,
            d_state=d_state,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            inner_rank=32,          # 논문 기본값 고정
            num_tokens=64,          # 논문 기본값 고정
            convffn_kernel_size=5,  # 논문 기본값 고정
            mlp_ratio=1.0,          # 논문 기본값 고정
            upsampler='pixelshuffledirect',
            resi_connection='1conv'
        )
        self.window_size = window_size

    def load_pretrained_weights(self, path):
        if not path:
            return
        print(f"[MambaIR] 가중치 로드 중: {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        # BasicSR 저장 방식 처리 (params_ema > params > dict)
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
            
        # 키 이름 정리 ('net_g.' 제거)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net_g.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict, strict=True)
        print("[MambaIR] ✓ 가중치 로드 완료")

    def forward_features(self, x):
        """
        [핵심] Arch 5-B용 Feature Extraction
        패딩 -> Feature 추출 -> 언패딩 과정을 캡슐화
        """
        # 1. Padding (Window Size 배수 맞추기)
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = (mod - h_ori % mod) % mod
        w_pad = (mod - w_ori % mod) % mod
        
        x_pad = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h_ori + h_pad, :]
        x_pad = torch.cat([x_pad, torch.flip(x_pad, [3])], 3)[:, :, :, :w_ori + w_pad]

        # 2. 마스크 생성 (모델 내부 함수 사용)
        attn_mask = self.model.calculate_mask([x_pad.shape[2], x_pad.shape[3]]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.model.relative_position_index_SA}
        
        # 3. Feature 추출
        feat = self.model.forward_features(x_pad, params)
        
        # 4. Unpadding
        feat = feat[..., :h_ori, :w_ori]
        
        return feat

    def forward(self, x):
        return self.model(x)
