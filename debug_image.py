import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from munch import Munch
import numpy as np
from core.utils import debug_image
from core.model import Generator, MappingNetwork, StyleEncoder
from core.wing import FAN
from PIL import Image

# モデルの定義
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / np.sqrt(2)

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / np.sqrt(2)
        return out

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 64
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0)
        )

        # Fixed repeat_num based on checkpoint
        repeat_num = 6

        # Encoder: Double channels up to max_conv_dim
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            dim_in = dim_out

        # Decoder: Specific channel progression to match checkpoint
        decoder_channels = [512, 512, 512, 256, 128, 64]
        dim_in = max_conv_dim  # Start from 512
        for dim_out in decoder_channels:
            self.decode.append(AdainResBlk(dim_in, dim_out, style_dim, w_hpf=w_hpf, upsample=True))
            dim_in = dim_out

        if w_hpf > 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) == 32 else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]
        return s

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=256):
        super().__init__()
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]  # shared.0

        # Dynamic repeat_num to achieve spatial_size=4
        repeat_num = int(np.log2(img_size)) - 2  # 512 -> 7, 256 -> 6
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]  # shared.1 to shared.7 (for 512)
            dim_in = dim_out

        spatial_size = img_size // (2 ** repeat_num)
        blocks += [nn.LeakyReLU(0.2)]  # shared.8
        blocks += [nn.Conv2d(dim_out, dim_out, kernel_size=spatial_size, stride=1, padding=0)]  # shared.9
        blocks += [nn.LeakyReLU(0.2)]  # shared.10
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]
        return s

class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat

@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)

@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat

@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = os.path.join(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    for psi in [0.5, 0.7, 1.0]:
        filename = os.path.join(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = os.path.join(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)

    # モデルの構造を確認
    print("Generatorの構造:")
    print(nets.generator)
    print("\nStyleEncoderの構造:")
    print(nets.style_encoder)

def load_models(args):
    # モデルの初期化
    nets = Munch()

    # Generatorの初期化（アーキテクチャを修正）
    nets.generator = Generator(
        img_size=args.img_size,
        style_dim=args.style_dim,
        w_hpf=args.w_hpf,
        max_conv_dim=args.max_conv_dim
    ).to(args.device)

    nets.mapping_network = MappingNetwork(
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        num_domains=args.num_domains
    ).to(args.device)

    nets.style_encoder = StyleEncoder(
        img_size=args.img_size,
        style_dim=args.style_dim,
        num_domains=args.num_domains,
        max_conv_dim=args.max_conv_dim
    ).to(args.device)

    print("Generatorの構造:")
    print(nets.generator)
    print("\nMappingNetworkの構造:")
    print(nets.mapping_network)
    print("\nStyleEncoderの構造:")
    print(nets.style_encoder)

    # FANモデルの初期化
    try:
        nets.fan = FAN(fname_pretrained=args.wing_path).to(args.device).eval()
        print("FANモデルを正常に読み込みました")
    except Exception as e:
        print(f"FANモデルの読み込みに失敗しました: {e}")
        print("FANモデルなしで実行します")
        nets.fan = None
        args.w_hpf = 0  # FANモデルがない場合はw_hpfを0に設定

    # チェックポイントの読み込み
    try:
        print(f"チェックポイントを読み込み中: {args.checkpoint_path}")
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {args.checkpoint_path}")

        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)

        # チェックポイントの内容を確認
        required_keys = ['generator', 'mapping_network', 'style_encoder']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(f"チェックポイントに必要なキーが不足しています: {missing_keys}")

        # モデルの重みを読み込み
        try:
            # チェックポイントの内容を確認
            print("チェックポイントのキー:", checkpoint.keys())
            print("Generatorのキー:", checkpoint['generator'].keys())
            print("StyleEncoderのキー:", checkpoint['style_encoder'].keys())

            # モデルの重みを読み込み
            nets.generator.load_state_dict(checkpoint['generator'], strict=False) # Keep strict=False
            nets.mapping_network.load_state_dict(checkpoint['mapping_network'])
            nets.style_encoder.load_state_dict(checkpoint['style_encoder'])
            print("チェックポイントを正常に読み込みました")

        except Exception as e:
            print(f"モデルの重みの読み込みに失敗しました: {e}")
            print("モデルのアーキテクチャを確認してください")
            return None

    except Exception as e:
        print(f"チェックポイントの読み込みに失敗しました: {e}")
        print("ランダムな重みで初期化します。")

    return nets

def main():
    # 引数の設定
    args = Munch()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.img_size = 256
    args.style_dim = 64
    args.latent_dim = 16
    args.num_domains = 23  # ドメインの総数
    args.w_hpf = 0
    args.num_outs_per_domain = 1
    args.max_conv_dim = 512
    args.repeat_num_offset = -2

    # チェックポイントのパスを設定
    checkpoint_dir = 'expr/checkpoints/foodimg512'
    checkpoint_path = os.path.join(checkpoint_dir, '200000_nets.ckpt')
    wing_path = 'expr/checkpoints/wing.ckpt'

    if not os.path.exists(checkpoint_dir):
        print(f"チェックポイントディレクトリが存在しません: {checkpoint_dir}")
        return

    args.checkpoint_path = checkpoint_path
    args.wing_path = wing_path
    args.sample_dir = 'expr/samples'
    os.makedirs(args.sample_dir, exist_ok=True)

    # モデルの読み込み
    nets = load_models(args)
    if nets is None:
        print("モデルの読み込みに失敗しました。")
        return

    

    # 画像の読み込みと前処理
    def preprocess_image(image_path, size=256):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]),
        ])
        return transform(img).unsqueeze(0)

        

    # x_src = torch.randn(batch_size, 3, args.img_size, args.img_size).to(args.device)
    # y_src = torch.zeros(batch_size).long().to(args.device)
    # x_ref = torch.randn(batch_size, 3, args.img_size, args.img_size).to(args.device)
    # y_ref = torch.ones(batch_size).long().to(args.device)
    
    try:
        # 入力画像のパス
        src_path = 'assets/representative/foodimg256/ref/rice/rice_003.jpg'
        ref_path = 'assets/representative/foodimg256/src/bibimbap/bibimbap_003.jpg'

        # 画像の読み込み
        x_src = preprocess_image(src_path).to(args.device)
        x_ref = preprocess_image(ref_path).to(args.device)

        # ドメインラベルの設定（バッチサイズを1に設定）
        batch_size = 1
        y_src = torch.zeros(batch_size).long().to(args.device)
        y_ref = torch.ones(batch_size).long().to(args.device)

        # 入力データのパッケージング
        inputs = Munch(x_src=x_src, y_src=y_src, x_ref=x_ref, y_ref=y_ref)

        # 変換の実行
        print("画像変換を開始します...")
        step = 14
        debug_image(nets, args, inputs, step)

        # 変換結果の保存
        print(f"変換結果が {args.sample_dir} に保存されました。")
        print("以下のファイルが生成されました：")
        print(f"- {step:06d}_cycle_consistency.jpg")
        print(f"- {step:06d}_latent_psi_0.5.jpg")
        print(f"- {step:06d}_latent_psi_0.7.jpg")
        print(f"- {step:06d}_latent_psi_1.0.jpg")
        print(f"- {step:06d}_reference.jpg")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == '__main__':
    main()