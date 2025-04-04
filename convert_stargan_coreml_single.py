# -*- coding: utf-8 -*-

# 既存のimportに以下を追加
import coremltools as ct
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import math
import argparse
from os.path import join as ospj
from pathlib import Path
from itertools import chain
import random
import time
import datetime
import json
import ffmpeg
import torchvision
import torchvision.utils as vutils
from tqdm import tqdm
from collections import namedtuple
from copy import deepcopy
from functools import partial
from munch import Munch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2
from skimage.filters import gaussian
import coremltools as ct
import torchvision.utils as vutils
from typing import Dict, List, Optional, Tuple, Union

# 型定義
Tensor = torch.Tensor
Module = nn.Module
DataLoader = data.DataLoader
OptionalTensor = Optional[Tensor]
OptionalModule = Optional[Module]

class Config:
    """設定を管理するクラス"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        # vars(args) だと Namespace の内部表現も含まれる可能性があるので明示的に取得
        arg_dict = {k: getattr(args, k) for k in dir(args) if not k.startswith('_')}
        return cls(**arg_dict)

# --- ここから Core ML 変換用のクラス定義 ---

class ImageProcessor:
    """画像処理関連のユーティリティクラス"""
    @staticmethod
    def denormalize(x: Tensor) -> Tensor:
        """正規化された画像を元に戻す"""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    @staticmethod
    def save_image(x: Tensor, ncol: int, filename: str) -> None:
        """画像を保存"""
        x = ImageProcessor.denormalize(x)
        vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

    @staticmethod
    def tensor2numpy255(tensor: Tensor) -> np.ndarray:
        """Tensorをnumpy配列に変換"""
        return ((tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype('uint8')

    @staticmethod
    def np2tensor(image: np.ndarray) -> Tensor:
        """numpy配列をTensorに変換"""
        return torch.FloatTensor(image).permute(2, 0, 1) / 255 * 2 - 1

class ModelManager:
    """モデル管理クラス"""
    def __init__(self, args: Config):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets, self.nets_ema = self._build_models()
        self._initialize_models()

    def _build_models(self) -> Tuple[Munch, Munch]:
        """モデルの構築"""
        generator = nn.DataParallel(Generator(self.args.img_size, self.args.style_dim, w_hpf=self.args.w_hpf))
        mapping_network = nn.DataParallel(MappingNetwork(self.args.latent_dim, self.args.style_dim, self.args.num_domains))
        style_encoder = nn.DataParallel(StyleEncoder(self.args.img_size, self.args.style_dim, self.args.num_domains))
        discriminator = nn.DataParallel(Discriminator(self.args.img_size, self.args.num_domains))
        
        nets = Munch(
            generator=generator,
            mapping_network=mapping_network,
            style_encoder=style_encoder,
            discriminator=discriminator
        )
        
        nets_ema = Munch(
            generator=copy.deepcopy(generator),
            mapping_network=copy.deepcopy(mapping_network),
            style_encoder=copy.deepcopy(style_encoder)
        )

        if self.args.w_hpf > 0:
            fan = nn.DataParallel(FAN(fname_pretrained=self.args.wing_path).eval())
            fan.get_heatmap = fan.module.get_heatmap
            nets.fan = fan
            nets_ema.fan = fan

        return nets, nets_ema

    def _initialize_models(self) -> None:
        """モデルの初期化"""
        for name, network in self.nets.items():
            if 'ema' not in name and 'fan' not in name:
                print(f'Initializing {name}...')
                network.apply(he_init)

    def save_checkpoint(self, step: int) -> None:
        """チェックポイントの保存"""
        for ckptio in self.ckptios:
            ckptio.save(step)

    def load_checkpoint(self, step: int) -> None:
        """チェックポイントの読み込み"""
        for ckptio in self.ckptios:
            ckptio.load(step)

class DataLoaderManager:
    """データローダー管理クラス"""
    def __init__(self, args: Config):
        self.args = args

    def get_test_loader(self, root: str, img_size: int = 256, batch_size: int = 32,
                       shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """テスト用データローダーの取得"""
        transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        dataset = ImageFolder(root, transform)
        return data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    def create_loaders(self) -> Munch:
        """必要なデータローダーの作成"""
        return Munch(
            src=self.get_test_loader(
                root=self.args.src_dir,
                img_size=self.args.img_size,
                batch_size=self.args.val_batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            ),
            ref=self.get_test_loader(
                root=self.args.ref_dir,
                img_size=self.args.img_size,
                batch_size=self.args.val_batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )
        )

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


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
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255

def get_preds_fromhm(hm):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)
    return preds


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(64, 64, True, True, 256, first_one,
                                     out_channels=256,
                                     kernel_size=1, stride=1, padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))
        self.add_module('b2_' + str(level), ConvBlock(256, 256))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))
        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)
        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class AddCoordsTh(nn.Module):
    def __init__(self, height=64, width=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.with_r = with_r
        self.with_boundary = with_boundary
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            x_coords = torch.arange(height).unsqueeze(1).expand(height, width).float()
            y_coords = torch.arange(width).unsqueeze(0).expand(height, width).float()
            x_coords = (x_coords / (height - 1)) * 2 - 1
            y_coords = (y_coords / (width - 1)) * 2 - 1
            coords = torch.stack([x_coords, y_coords], dim=0)  # (2, height, width)

            if self.with_r:
                rr = torch.sqrt(torch.pow(x_coords, 2) + torch.pow(y_coords, 2))  # (height, width)
                rr = (rr / torch.max(rr)).unsqueeze(0)
                coords = torch.cat([coords, rr], dim=0)

            self.coords = coords.unsqueeze(0).to(device)  # (1, 2 or 3, height, width)
            self.x_coords = x_coords.to(device)
            self.y_coords = y_coords.to(device)

    def forward(self, x, heatmap=None):
        """
        x: (batch, c, x_dim, y_dim)
        """
        coords = self.coords.repeat(x.size(0), 1, 1, 1)

        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)
            zero_tensor = torch.zeros_like(self.x_coords)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, self.x_coords, zero_tensor).to(zero_tensor.device)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, self.y_coords, zero_tensor).to(zero_tensor.device)
            coords = torch.cat([coords, xx_boundary_channel, yy_boundary_channel], dim=1)

        x_and_coords = torch.cat([x, coords], dim=1)
        return x_and_coords


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, height, width, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(height, width, with_r, with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.BatchNorm2d(in_planes),
                                            nn.ReLU(True),
                                            nn.Conv2d(in_planes, out_planes, 1, 1, bias=False))

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


class FAN(nn.Module):
    def __init__(self, num_modules=1, end_relu=False, num_landmarks=98, fname_pretrained=None):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.end_relu = end_relu

        # Base part
        self.conv1 = CoordConvTh(256, 256, True, False,
                                 in_channels=3, out_channels=64,
                                 kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        self.add_module('m0', HourGlass(1, 4, 256, first_one=True))
        self.add_module('top_m_0', ConvBlock(256, 256))
        self.add_module('conv_last0', nn.Conv2d(256, 256, 1, 1, 0))
        self.add_module('bn_end0', nn.BatchNorm2d(256))
        self.add_module('l0', nn.Conv2d(256, num_landmarks+1, 1, 1, 0))

        if fname_pretrained is not None:
            self.load_pretrained_weights(fname_pretrained)

    def load_pretrained_weights(self, fname):
        if torch.cuda.is_available():
            checkpoint = torch.load(fname)
        else:
            checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        model_weights = self.state_dict()
        model_weights.update({k: v for k, v in checkpoint['state_dict'].items()
                              if k in model_weights})
        self.load_state_dict(model_weights)

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        boundary_channels = []
        tmp_out = None
        ll, boundary_channel = self._modules['m0'](x, tmp_out)
        ll = self._modules['top_m_0'](ll)
        ll = F.relu(self._modules['bn_end0']
                    (self._modules['conv_last0'](ll)), True)

        # Predict heatmaps
        tmp_out = self._modules['l0'](ll)
        if self.end_relu:
            tmp_out = F.relu(tmp_out)  # HACK: Added relu
        outputs.append(tmp_out)
        boundary_channels.append(boundary_channel)
        return outputs, boundary_channels

    @torch.no_grad()
    def get_heatmap(self, x, b_preprocess=True):
        ''' outputs 0-1 normalized heatmap '''
        x = F.interpolate(x, size=256, mode='bilinear')
        x_01 = x*0.5 + 0.5
        outputs, _ = self(x_01)
        heatmaps = outputs[-1][:, :-1, :, :]
        scale_factor = x.size(2) // heatmaps.size(2)
        if b_preprocess:
            heatmaps = F.interpolate(heatmaps, scale_factor=scale_factor,
                                     mode='bilinear', align_corners=True)
            heatmaps = preprocess(heatmaps)
        return heatmaps

    @torch.no_grad()
    def get_landmark(self, x):
        ''' outputs landmarks of x.shape '''
        heatmaps = self.get_heatmap(x, b_preprocess=False)
        landmarks = []
        for i in range(x.size(0)):
            pred_landmarks = get_preds_fromhm(heatmaps[i].cpu().unsqueeze(0))
            landmarks.append(pred_landmarks)
        scale_factor = x.size(2) // heatmaps.size(2)
        landmarks = torch.cat(landmarks) * scale_factor
        return landmarks


# ========================== #
#   Align related functions  #
# ========================== #


def tensor2numpy255(tensor):
    """Converts torch tensor to numpy array."""
    return ((tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype('uint8')


def np2tensor(image):
    """Converts numpy array to torch tensor."""
    return torch.FloatTensor(image).permute(2, 0, 1) / 255 * 2 - 1


class FaceAligner():
    def __init__(self, fname_wing, fname_celeba_mean, output_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fan = FAN(fname_pretrained=fname_wing).to(self.device).eval()
        scale = output_size // 256
        self.CELEB_REF = np.float32(np.load(fname_celeba_mean)['mean']) * scale
        self.xaxis_ref = landmarks2xaxis(self.CELEB_REF)
        self.output_size = output_size

    def align(self, imgs, output_size=256):
        ''' imgs = torch.CUDATensor of BCHW '''
        imgs = imgs.to(self.device)
        landmarkss = self.fan.get_landmark(imgs).cpu().numpy()
        for i, (img, landmarks) in enumerate(zip(imgs, landmarkss)):
            img_np = tensor2numpy255(img)
            img_np, landmarks = pad_mirror(img_np, landmarks)
            transform = self.landmarks2mat(landmarks)
            rows, cols, _ = img_np.shape
            rows = max(rows, self.output_size)
            cols = max(cols, self.output_size)
            aligned = cv2.warpPerspective(img_np, transform, (cols, rows), flags=cv2.INTER_LANCZOS4)
            imgs[i] = np2tensor(aligned[:self.output_size, :self.output_size, :])
        return imgs

    def landmarks2mat(self, landmarks):
        T_origin = points2T(landmarks, 'from')
        xaxis_src = landmarks2xaxis(landmarks)
        R = vecs2R(xaxis_src, self.xaxis_ref)
        S = landmarks2S(landmarks, self.CELEB_REF)
        T_ref = points2T(self.CELEB_REF, 'to')
        matrix = np.dot(T_ref, np.dot(S, np.dot(R, T_origin)))
        return matrix


def points2T(point, direction):
    point_mean = point.mean(axis=0)
    T = np.eye(3)
    coef = -1 if direction == 'from' else 1
    T[:2, 2] = coef * point_mean
    return T


def landmarks2eyes(landmarks):
    idx_left = np.array(list(range(60, 67+1)) + [96])
    idx_right = np.array(list(range(68, 75+1)) + [97])
    left = landmarks[idx_left]
    right = landmarks[idx_right]
    return left.mean(axis=0), right.mean(axis=0)


def landmarks2mouthends(landmarks):
    left = landmarks[76]
    right = landmarks[82]
    return left, right


def rotate90(vec):
    x, y = vec
    return np.array([y, -x])


def landmarks2xaxis(landmarks):
    eye_left, eye_right = landmarks2eyes(landmarks)
    mouth_left, mouth_right = landmarks2mouthends(landmarks)
    xp = eye_right - eye_left  # x' in pggan
    eye_center = (eye_left + eye_right) * 0.5
    mouth_center = (mouth_left + mouth_right) * 0.5
    yp = eye_center - mouth_center
    xaxis = xp - rotate90(yp)
    return xaxis / np.linalg.norm(xaxis)


def vecs2R(vec_x, vec_y):
    vec_x = vec_x / np.linalg.norm(vec_x)
    vec_y = vec_y / np.linalg.norm(vec_y)
    c = np.dot(vec_x, vec_y)
    s = np.sqrt(1 - c * c) * np.sign(np.cross(vec_x, vec_y))
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    return R


def landmarks2S(x, y):
    x_mean = x.mean(axis=0).squeeze()
    y_mean = y.mean(axis=0).squeeze()
    # vectors = mean -> each point
    x_vectors = x - x_mean
    y_vectors = y - y_mean

    x_norms = np.linalg.norm(x_vectors, axis=1)
    y_norms = np.linalg.norm(y_vectors, axis=1)

    indices = [96, 97, 76, 82]  # indices for eyes, lips
    scale = (y_norms / x_norms)[indices].mean()

    S = np.eye(3)
    S[0, 0] = S[1, 1] = scale
    return S


def pad_mirror(img, landmarks):
    H, W, _ = img.shape
    img = np.pad(img, ((H//2, H//2), (W//2, W//2), (0, 0)), 'reflect')
    small_blurred = gaussian(cv2.resize(img, (W, H)), H//100, multichannel=True)
    blurred = cv2.resize(small_blurred, (W * 2, H * 2)) * 255

    H, W, _ = img.shape
    coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    weight_y = np.clip(coords[0] / (H//4), 0, 1)
    weight_x = np.clip(coords[1] / (H//4), 0, 1)
    weight_y = np.minimum(weight_y, np.flip(weight_y, axis=0))
    weight_x = np.minimum(weight_x, np.flip(weight_x, axis=1))
    weight = np.expand_dims(np.minimum(weight_y, weight_x), 2)**4
    img = img * weight + blurred * (1 - weight)
    landmarks += np.array([W//4, H//4])
    return img, landmarks


def align_faces(args, input_dir, output_dir):
    import os
    from torchvision import transforms
    from PIL import Image
    from core.utils import save_image

    aligner = FaceAligner(args.wing_path, args.lm_path, args.img_size)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    fnames = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    fnames.sort()
    for fname in fnames:
        image = Image.open(os.path.join(input_dir, fname)).convert('RGB')
        x = transform(image).unsqueeze(0)
        x_aligned = aligner.align(x)
        save_image(x_aligned, 1, filename=os.path.join(output_dir, fname))
        print('Saved the aligned image to %s...' % fname)


# ========================== #
#   Mask related functions   #
# ========================== #


def normalize(x, eps=1e-6):
    """Apply min-max normalization."""
    x = x.contiguous()
    N, C, H, W = x.size()
    x_ = x.view(N*C, -1)
    max_val = torch.max(x_, dim=1, keepdim=True)[0]
    min_val = torch.min(x_, dim=1, keepdim=True)[0]
    x_ = (x_ - min_val) / (max_val - min_val + eps)
    out = x_.view(N, C, H, W)
    return out


def truncate(x, thres=0.1):
    """Remove small values in heatmaps."""
    return torch.where(x < thres, torch.zeros_like(x), x)


def resize(x, p=2):
    """Resize heatmaps."""
    return x**p


def shift(x, N):
    """Shift N pixels up or down."""
    up = N >= 0
    N = abs(N)
    _, _, H, W = x.size()
    head = torch.arange(N)
    tail = torch.arange(H-N)

    if up:
        head = torch.arange(H-N)+N
        tail = torch.arange(N)
    else:
        head = torch.arange(N) + (H-N)
        tail = torch.arange(H-N)

    # permutation indices
    perm = torch.cat([head, tail]).to(x.device)
    out = x[:, :, perm, :]
    return out


IDXPAIR = namedtuple('IDXPAIR', 'start end')
index_map = Munch(chin=IDXPAIR(0 + 8, 33 - 8),
                  eyebrows=IDXPAIR(33, 51),
                  eyebrowsedges=IDXPAIR(33, 46),
                  nose=IDXPAIR(51, 55),
                  nostrils=IDXPAIR(55, 60),
                  eyes=IDXPAIR(60, 76),
                  lipedges=IDXPAIR(76, 82),
                  lipupper=IDXPAIR(77, 82),
                  liplower=IDXPAIR(83, 88),
                  lipinner=IDXPAIR(88, 96))
OPPAIR = namedtuple('OPPAIR', 'shift resize')


def preprocess(x):
    """Preprocess 98-dimensional heatmaps."""
    N, C, H, W = x.size()
    x = truncate(x)
    x = normalize(x)

    sw = H // 256
    operations = Munch(chin=OPPAIR(0, 3),
                       eyebrows=OPPAIR(-7*sw, 2),
                       nostrils=OPPAIR(8*sw, 4),
                       lipupper=OPPAIR(-8*sw, 4),
                       liplower=OPPAIR(8*sw, 4),
                       lipinner=OPPAIR(-2*sw, 3))

    for part, ops in operations.items():
        start, end = index_map[part]
        x[:, start:end] = resize(shift(x[:, start:end], ops.shift), ops.resize)

    zero_out = torch.cat([torch.arange(0, index_map.chin.start),
                          torch.arange(index_map.chin.end, 33),
                          torch.LongTensor([index_map.eyebrowsedges.start,
                                            index_map.eyebrowsedges.end,
                                            index_map.lipedges.start,
                                            index_map.lipedges.end])])
    x[:, zero_out] = 0

    start, end = index_map.nose
    x[:, start+1:end] = shift(x[:, start+1:end], 4*sw)
    x[:, start:end] = resize(x[:, start:end], 1)

    start, end = index_map.eyes
    x[:, start:end] = resize(x[:, start:end], 1)
    x[:, start:end] = resize(shift(x[:, start:end], -8), 3) + \
        shift(x[:, start:end], -24)

    # Second-level mask
    x2 = deepcopy(x)
    x2[:, index_map.chin.start:index_map.chin.end] = 0  # start:end was 0:33
    x2[:, index_map.lipedges.start:index_map.lipinner.end] = 0  # start:end was 76:96
    x2[:, index_map.eyebrows.start:index_map.eyebrows.end] = 0  # start:end was 33:51

    x = torch.sum(x, dim=1, keepdim=True)  # (N, 1, H, W)
    x2 = torch.sum(x2, dim=1, keepdim=True)  # mask without faceline and mouth

    x[x != x] = 0  # set nan to zero
    x2[x != x] = 0  # set nan to zero
    return x.clamp_(0, 1), x2.clamp_(0, 1)

# from core.solver import Solver
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
        return x / math.sqrt(2)  # unit variance


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
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


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

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
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
                mask = masks[0] if x.size(2) in [32] else masks[1]
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
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
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
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out

class CheckpointIO(object):
    def __init__(self, fname_template, data_parallel=False, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            if self.data_parallel:
                outdict[name] = module.module.state_dict()
            else:
                outdict[name] = module.state_dict()
                        
        torch.save(outdict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
            
        for name, module in self.module_dict.items():
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name], strict=False)
            else:
                module.load_state_dict(module_dict[name], strict=False)

class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
    
def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = nn.DataParallel(FAN(fname_pretrained=args.wing_path).eval())
        fan.get_heatmap = fan.module.get_heatmap
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema

def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # real
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # fake
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    loss = loss_real + loss_fake + args.lambda_reg * loss_reg

    return loss, Munch(real=loss_real.item(),
                      fake=loss_fake.item(),
                      reg=loss_reg.item())

def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs, x_refs, masks=None):
    batch_size = x_real.size(0)
    s_trg = nets.mapping_network(z_trgs[0], y_trg)

    # adversarial loss
    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # diversity sensitive loss
    if z_trgs is not None:
        z_trg2 = z_trgs[1]
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
        x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
        x_fake2 = x_fake2.detach()
        out = nets.discriminator(x_fake2, y_trg)
        loss_ds = torch.mean(out)
    else:
        loss_ds = 0

    # reference-based reconstruction loss
    if x_refs is not None:
        x_ref = x_refs[0]
        s_ref = nets.style_encoder(x_ref, y_trg)
        x_ref2 = nets.generator(x_real, s_ref, masks=masks)
        loss_ref = torch.mean(torch.abs(x_ref2 - x_ref))
    else:
        loss_ref = 0

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc \
        + args.lambda_ref * loss_ref

    return loss, Munch(adv=loss_adv.item(),
                      sty=loss_sty.item(),
                      ds=loss_ds.item(),
                      cyc=loss_cyc.item(),
                      ref=loss_ref.item())

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1).mean()
    return reg

def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def check_checkpoint(args):
    # netsのチェックポイントを読み込む
    checkpoint_path = ospj(args.checkpoint_dir, f'{args.resume_iter}_nets_ema.ckpt')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    print("Checkpoint structure:")
    for key in checkpoint.keys():
        print(f"- {key}")
    
    # discriminatorの出力層のサイズを確認
    if 'discriminator' in checkpoint:
        discriminator = checkpoint['discriminator']
        for key, value in discriminator.items():
            if 'weight' in key and 'conv' in key:
                print(f"\nDiscriminator output size: {value.shape[0]}")
                break


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)

class StyleEncoderWrapper(nn.Module):
    """
    StyleEncoderをラップし、参照画像から全ドメインのスタイルコードを出力する。
    Core ML (trace-based) 変換のため。
    """
    def __init__(self, style_encoder_module: Module, num_domains: int, style_dim: int):
        super().__init__()
        # DataParallelがかかっている可能性があるので .module を取る
        se_module = style_encoder_module.module if isinstance(style_encoder_module, nn.DataParallel) else style_encoder_module
        self.shared = se_module.shared
        self.unshared = se_module.unshared
        self.num_domains = num_domains
        self.style_dim = style_dim

        # unshared モジュールの数をチェック
        if len(self.unshared) != self.num_domains:
             print(f"Warning: Number of unshared layers ({len(self.unshared)}) in StyleEncoder does not match num_domains ({self.num_domains}).")
             # 必要に応じてエラー処理または調整

    def forward(self, x: Tensor) -> Tensor:
        h = self.shared(x)
        h = h.view(h.size(0), -1) # Flatten
        styles = []
        # JIT traceのために、num_domainsが固定値であることを利用
        for i in range(self.num_domains):
             # unsharedモジュールが存在するか確認
             if i < len(self.unshared):
                 styles.append(self.unshared[i](h))
             else:
                 # ドメイン数とモジュール数が合わない場合のフォールバック (例: ゼロベクトル)
                 print(f"Warning: Missing unshared layer for domain {i}. Returning zeros.")
                 styles.append(torch.zeros(h.size(0), self.style_dim, device=h.device))

        # (batch, num_domains, style_dim) を返す
        all_styles = torch.stack(styles, dim=1)
        return all_styles

class StarGANv2CoreML(nn.Module):
    """
    StyleEncoderWrapperとGeneratorを結合し、Core ML変換可能な単一モジュールにする。
    入力: ソース画像, 参照画像, 参照ドメインインデックス
    出力: 生成画像
    """
    def __init__(self, style_encoder_wrapper: StyleEncoderWrapper, generator: Module, style_dim: int):
        super().__init__()
        self.style_encoder_wrapper = style_encoder_wrapper
        # GeneratorもDataParallelがかかっている可能性
        self.generator = generator.module if isinstance(generator, nn.DataParallel) else generator
        self.style_dim = style_dim

    def forward(self, source_image: Tensor, reference_image: Tensor, reference_domain_index: Tensor) -> Tensor:
        # reference_domain_index は shape [1] の Int64 Tensor (例: torch.tensor([5], dtype=torch.int64))

        # 全ドメインのスタイルコードを取得 (batch, num_domains, style_dim)
        # バッチサイズは1と仮定
        all_styles = self.style_encoder_wrapper(reference_image) # [1, num_domains, style_dim]

        # 指定されたドメインのスタイルコードを選択 (batch, style_dim)
        # reference_domain_index から整数インデックスを取得
        # バッチサイズ1、ドメインインデックスが単一の場合:
        # domain_idx = reference_domain_index[0] # スカラーTensor
        # gather を使用してインデックス選択 (Trace可能)
        # domain_idx を gather 用の形状に変換: [1, 1, 1]
        gather_index = reference_domain_index.view(1, 1, 1).expand(1, 1, self.style_dim)
        # all_styles [1, num_domains, style_dim], gather_index [1, 1, style_dim]
        # dim=1 (num_domains次元) に沿ってインデックス選択
        style_code = torch.gather(all_styles, 1, gather_index).squeeze(1) # [1, style_dim]

        # Generatorのforwardは (x, s, masks=None) を取る
        # masks は w_hpf=0 なので不要
        generated_image = self.generator(source_image, style_code, masks=None)
        return generated_image

# --- ここまで Core ML 変換用のクラス定義 ---

# --- ここから Core ML 変換関数 ---

def convert_to_coreml(args: Config, nets_ema: Munch):
    """PyTorchモデルをCore ML (.mlpackage) に変換する"""
    print("\nStarting Core ML model conversion...")

    img_size = args.img_size
    style_dim = args.style_dim
    num_domains = args.num_domains
    # 結果ディレクトリが存在しない場合は作成
    os.makedirs(args.result_dir, exist_ok=True)
    coreml_model_path = ospj(args.result_dir, f"StarGANv2_{args.img_size}.mlpackage")


    # 1. モデルの準備 (推論モード、DataParallel解除)
    try:
        style_encoder = nets_ema.style_encoder.eval()
        generator = nets_ema.generator.eval()
        se_module = style_encoder.module if isinstance(style_encoder, nn.DataParallel) else style_encoder
        gen_module = generator.module if isinstance(generator, nn.DataParallel) else generator
    except AttributeError as e:
        print(f"Error accessing models from nets_ema: {e}")
        print("Make sure nets_ema contains 'style_encoder' and 'generator'.")
        return

    # 2. ラッパーモデルのインスタンス化
    try:
        style_encoder_wrapper = StyleEncoderWrapper(se_module, num_domains, style_dim)
        coreml_compatible_model = StarGANv2CoreML(style_encoder_wrapper, gen_module, style_dim)
        coreml_compatible_model.eval()
        print("CoreML wrapper model created successfully.")
    except Exception as e:
        print(f"Error during model wrapper initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. ダミー入力の作成
    dummy_source_image = torch.randn(1, 3, img_size, img_size)
    dummy_reference_image = torch.randn(1, 3, img_size, img_size)
    dummy_reference_domain_index = torch.tensor([0], dtype=torch.int64)

    print(f"Dummy input shapes:")
    print(f"  Source Image:      {dummy_source_image.shape}")
    print(f"  Reference Image:   {dummy_reference_image.shape}")
    print(f"  Domain Index:      {dummy_reference_domain_index.shape} (dtype={dummy_reference_domain_index.dtype})")


    # 4. TorchScriptモデルへの変換 (Trace)
    print("Tracing model with torch.jit.trace...")
    try:
        traced_model = torch.jit.trace(coreml_compatible_model,
                                      (dummy_source_image, dummy_reference_image, dummy_reference_domain_index),
                                      strict=False)
        print("Model traced successfully.")
    except Exception as e:
        print(f"Error during torch.jit.trace: {e}")
        import traceback
        traceback.print_exc()
        print("Consider checking the model structure or trying torch.jit.script if trace fails.")
        return


    # 5. Core MLモデルへの変換
    # 入力画像の正規化
    image_scale = 2.0 / 255.0
    image_bias = [-1.0, -1.0, -1.0] # RGB

    source_image_input = ct.ImageType(name="source_image",
                                      shape=(1, 3, img_size, img_size),
                                      scale=image_scale, bias=image_bias,
                                      color_layout=ct.colorlayout.RGB,
                                      channel_first=True)

    reference_image_input = ct.ImageType(name="reference_image",
                                         shape=(1, 3, img_size, img_size),
                                         scale=image_scale, bias=image_bias,
                                         color_layout=ct.colorlayout.RGB,
                                         channel_first=True)

    domain_index_input = ct.TensorType(name="reference_domain_index",
                                        shape=(1,),
                                        dtype=np.int64) # coremltoolsがint32にキャストする可能性あり

    # --- 修正: 出力は TensorType のままにする ---
    # ct.convert で指定する出力名 (例: "generated_image_raw")
    output_tensor_name = "generated_image_tensor"
    generated_output = ct.TensorType(name=output_tensor_name)
    # --- 修正ここまで ---

    print("Converting to Core ML format (mlprogram)...")
    try:
        coreml_model = ct.convert(
            traced_model,
            inputs=[source_image_input, reference_image_input, domain_index_input],
            # --- 修正: TensorType の出力を指定 ---
            outputs=[generated_output],
            # --- 修正ここまで ---
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15, # または適切なターゲット
            compute_units=ct.ComputeUnit.ALL,
        )
        print("Core ML conversion successful.")

        # --- 修正: ImageTypeへの変更と後処理設定部分を削除 ---
        # spec = coreml_model.get_spec()
        # ... (spec編集のコード全体を削除) ...
        # coreml_model = ct.models.MLModel(spec, weights_dir=coreml_model.weights_dir) # この行も不要
        # --- 修正ここまで ---

        # 6. メタデータの設定 (coreml_model オブジェクトに直接設定)
        coreml_model.short_description = f"StarGAN v2 Image Style Transfer (Reference-based, {img_size}x{img_size})"
        coreml_model.author = "Generated by script based on StarGAN v2"
        coreml_model.license = "Check original StarGAN v2 license"
        coreml_model.version = f"{args.resume_iter}"

        # 入力の説明
        coreml_model.input_description["source_image"] = f"Source image ({img_size}x{img_size} RGB) normalized to [-1, 1]"
        coreml_model.input_description["reference_image"] = f"Reference image ({img_size}x{img_size} RGB) normalized to [-1, 1]"
        coreml_model.input_description["reference_domain_index"] = f"Index of the target domain (Int Tensor[1], 0 to {num_domains-1})" # int64 or int32

        # --- 修正: 出力の説明を Tensor であることを示すように変更 ---
        coreml_model.output_description[output_tensor_name] = f"Generated image as Tensor (1, 3, {img_size}, {img_size}) in RGB order, normalized to [-1, 1]. Apply postprocessing: (output + 1) / 2 * 255."
        # --- 修正ここまで ---

        # 7. モデルの保存
        coreml_model.save(coreml_model_path)
        print(f"Core ML model saved to: {coreml_model_path}")
        # アプリ側での後処理が必要であることを明記
        print("Important: The Core ML model output is a normalized tensor.")
        print("           You need to apply post-processing in your application:")
        print("           pixel_value = (tensor_output + 1.0) / 2.0 * 255.0")


    except Exception as e:
        print(f"\nError during Core ML conversion or saving: {e}")
        import traceback
        traceback.print_exc()
        print("\nConversion failed.")

# --- ここまで Core ML 変換関数 ---


# Solverクラスも既存のまま使用
class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # build_modelを呼び出してモデルを構築・初期化
        self.nets, self.nets_ema = build_model(args)
        # solverの属性としてモデルを設定 (DataParallelラップされたまま)
        for name, module in self.nets.items():
            # print_network(module, name) # 必要なら表示
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        # build_model内で初期化されるので、ここでの初期化は不要かも
        # for name, network in self.nets.items():
        #     if 'ema' not in name and 'fan' not in name:
        #         print('Initializing %s...' % name)
        #         network.apply(he_init)

        # CheckpointIOの設定 (ロード専用)
        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)


    def _load_checkpoint(self, step):
        print(f"Loading EMA checkpoint for step {step}...")
        # nets_ema のみをロード
        ckptio_ema = CheckpointIO(ospj(self.args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)
        try:
            ckptio_ema.load(step)
            print("EMA Checkpoint loaded successfully.")
        except FileNotFoundError:
            print(f"Error: EMA Checkpoint file not found for step {step} at {self.args.checkpoint_dir}")
            raise
        except Exception as e:
            print(f"Error loading EMA checkpoint: {e}")
            raise

    @torch.no_grad()
    def sample(self, loaders):
        """PyTorchでのサンプル画像生成（オプション）"""
        args = self.args
        nets_ema = self.nets_ema # Solverの属性を使う
        os.makedirs(args.result_dir, exist_ok=True)
        # チェックポイントは main でロード済みのはず

        print("Fetching data for PyTorch sampling...")
        try:
            # データローダーから1バッチ取得
            fetcher_src = InputFetcher(loaders.src, None, args.latent_dim, 'test')
            fetcher_ref = InputFetcher(loaders.ref, None, args.latent_dim, 'test')
            src = next(fetcher_src)
            ref = next(fetcher_ref)
        except StopIteration:
            print("Error: Could not fetch data from loaders for sampling.")
            return
        except Exception as e:
            print(f"Error fetching data: {e}")
            return

        fname = ospj(args.result_dir, 'reference_pytorch_sample.jpg')
        print('Working on {}...'.format(fname))
        # translate_using_reference を呼び出す
        # この関数は nets (Munch) を期待するが、nets_ema を渡す
        translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)
        print(f"PyTorch sample saved to {fname}")

def main(args: Config): # Configオブジェクトを受け取るように変更
    print("Configuration:")
    print(json.dumps(args.__dict__, indent=2)) # Configオブジェクトの内容を表示

    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # Solverのインスタンス化（モデル構築と初期化を含む）
    solver = Solver(args) # argsはConfigオブジェクト

    # チェックポイントのロード
    try:
        solver._load_checkpoint(args.resume_iter)
    except Exception as e:
        print(f"Failed to load checkpoint for iter {args.resume_iter}. Exiting.")
        return

    # --- CoreMLモデルの作成 ---
    # nets_ema は solver インスタンスが保持している
    convert_to_coreml(args, solver.nets_ema)
    print("\nCore ML conversion process finished.")

    # --- Core ML 推論テストの実行 ---
    coreml_model_path = ospj(args.result_dir, f"StarGANv2_{args.img_size}.mlpackage")
    print(f"Core ML 推論テストの実行中: {coreml_model_path}")
    if os.path.exists(coreml_model_path):
         # src_dir と ref_dir が指定されている場合のみ実行
        if args.src_dir and args.ref_dir:
            run_coreml_inference_test(args, coreml_model_path)
        else:
            print("\nCore ML 推論テストをスキップ: --src_dir または --ref_dir が指定されていません。")
    else:
        print(f"\nCore ML 推論テストをスキップ: モデルファイルが見つかりません {coreml_model_path}")

    # --- サンプル生成の実行 (オプション) ---
    run_pytorch_sampling = True # Falseにすればスキップ
    if run_pytorch_sampling:
        print("\nRunning PyTorch sampling for comparison...")
        # データローダーの設定 (サンプル生成用)
        try:
            # src_dirとref_dirのサブディレクトリ数をチェック
            if not os.path.isdir(args.src_dir) or not os.path.isdir(args.ref_dir):
                 print(f"Error: src_dir '{args.src_dir}' or ref_dir '{args.ref_dir}' not found.")
                 return
            src_subdirs = len(subdirs(args.src_dir))
            ref_subdirs = len(subdirs(args.ref_dir))
            if src_subdirs == 0 or ref_subdirs == 0:
                 print(f"Warning: No subdirectories found in src_dir or ref_dir. Assuming single domain structure.")
                 # num_domainsが1より大きい場合はエラーにするか、処理を継続するか選択
                 if args.num_domains > 1:
                      print(f"Error: num_domains is {args.num_domains}, but image directories lack domain subfolders.")
                      # return # エラーにする場合
            elif src_subdirs != args.num_domains:
                 print(f"Error: Number of subdirs in src_dir ({src_subdirs}) must match num_domains ({args.num_domains})")
                 return
            elif ref_subdirs != args.num_domains:
                print(f"Error: Number of subdirs in ref_dir ({ref_subdirs}) must match num_domains ({args.num_domains})")
                return

            loaders = Munch(src=get_test_loader(root=args.src_dir,
                                                img_size=args.img_size,
                                                batch_size=min(4, args.val_batch_size), # サンプル用に少量
                                                shuffle=False, # サンプル時はFalse推奨
                                                num_workers=args.num_workers),
                            ref=get_test_loader(root=args.ref_dir,
                                                img_size=args.img_size,
                                                batch_size=min(4, args.val_batch_size), # サンプル用に少量
                                                shuffle=False, # サンプル時はFalse推奨
                                                num_workers=args.num_workers))
            solver.sample(loaders)
        except Exception as e:
            print(f"\nError during PyTorch sampling: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nPyTorch sampling skipped.")

def save_coreml_output(output_dict: Dict[str, np.ndarray], output_key: str, filename: str):
    """Core MLモデルのテンソル出力を画像として保存する"""
    if output_key not in output_dict:
        print(f"Error: 出力キー '{output_key}' が推論結果に見つかりません。")
        return

    # Core MLの出力は通常 (batch, C, H, W) の NumPy 配列
    output_tensor_np = output_dict[output_key]
    print(f"Core ML Output shape: {output_tensor_np.shape}, dtype: {output_tensor_np.dtype}, min: {output_tensor_np.min()}, max: {output_tensor_np.max()}")


    # NumPy配列をPyTorchテンソルに変換
    output_tensor = torch.from_numpy(output_tensor_np)

    # 非正規化: (tensor + 1) / 2
    # 既存のdenormalize関数を使用するか、ここで定義
    # (例)
    def denormalize(tensor: Tensor) -> Tensor:
        """正規化された画像を元に戻す ([ -1, 1 ] -> [ 0, 1 ])"""
        out = (tensor + 1.0) / 2.0
        return out.clamp_(0.0, 1.0) # 念のため[0, 1]にクリップ

    # バッチの最初の要素を取得して非正規化
    if output_tensor.dim() == 4 and output_tensor.shape[0] == 1:
        denormalized_tensor = denormalize(output_tensor[0])
    else:
        # バッチサイズが1でない、または予期しない形状の場合の処理 (必要に応じて調整)
        print(f"Warning: 予期しない出力テンソルの形状 {output_tensor.shape}。最初の要素を使用します。")
        denormalized_tensor = denormalize(output_tensor[0] if output_tensor.dim() > 2 else output_tensor)

    # 画像を保存
    try:
        vutils.save_image(denormalized_tensor, filename)
        print(f"Core ML 出力画像を保存しました: {filename}")
    except Exception as e:
        print(f"Error: Core ML 出力画像の保存中にエラーが発生しました: {e}")

def run_coreml_inference_test(args: Config, coreml_model_path: str):
    """Core MLモデルをロードし、サンプルデータで推論を実行する"""
    print(f"\n--- Core ML 推論テスト開始 ---")
    print(f"Core ML モデルをロード中: {coreml_model_path}")

    if not os.path.exists(coreml_model_path):
        print(f"Error: Core ML モデルファイルが見つかりません: {coreml_model_path}")
        return

    try:
        # モデルをロード
        mlmodel = ct.models.MLModel(coreml_model_path)
        print("Core ML モデルのロード成功。")

        # 入出力情報を表示 (確認用)
        print("モデル入力情報:", mlmodel.input_description)
        print("モデル出力情報:", mlmodel.output_description)

        # 入力データの準備 (PyTorchサンプリングと同様だが、PIL/NumPyを使用)
        if not args.src_dir or not args.ref_dir:
            print("Error: Core ML 推論テストには --src_dir と --ref_dir が必要です。")
            return

        # リサイズ変換を定義 (訓練時と合わせた潰しリサイズ)
        img_size = args.img_size
        # predictが期待する形式 (通常はPIL Image) の直前までの変換
        preprocess_pil = transforms.Compose([
            transforms.Resize([img_size, img_size], interpolation=transforms.InterpolationMode.BILINEAR), # 潰しリサイズ
        ])

        # ソース画像を1枚ロード
        src_domain_folders = [d for d in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, d))]
        src_domain_folders.sort() # 順序を安定させる
        if not src_domain_folders:
             print(f"Error: src_dir にドメインサブフォルダが見つかりません: {args.src_dir}")
             return
        src_img_path = None
        # 最初のドメインの最初の画像を使用 (例)
        src_folder_path = os.path.join(args.src_dir, src_domain_folders[0])
        src_img_files = [f for f in os.listdir(src_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if src_img_files:
             src_img_files.sort()
             src_img_path = os.path.join(src_folder_path, src_img_files[0])
        if not src_img_path:
            print(f"Error: src_dir のサブフォルダに画像ファイルが見つかりません。")
            return
        print(f"ソース画像をロード中: {src_img_path}")
        src_img_pil = Image.open(src_img_path).convert('RGB')
        src_img_pil_resized = preprocess_pil(src_img_pil) # リサイズのみ適用

        # 参照画像を1枚ロードし、そのドメインインデックスを取得
        ref_domain_folders = [d for d in os.listdir(args.ref_dir) if os.path.isdir(os.path.join(args.ref_dir, d))]
        if not ref_domain_folders:
             print(f"Error: ref_dir にドメインサブフォルダが見つかりません: {args.ref_dir}")
             return
        ref_domain_folders.sort() # Swiftの 'domains' 配列と順序を合わせる

        # 例: 最初のドメインを参照として使用
        target_domain_name = ref_domain_folders[0]
        try:
            # ドメイン名からインデックスを取得
            target_domain_index = ref_domain_folders.index(target_domain_name)
        except ValueError:
            # Swiftのdomainsリストに含まれる名前と一致しない可能性に対処
            print(f"Error: ドメイン名 '{target_domain_name}' が ref_dir のフォルダリスト内に見つかりません。Swiftのdomains配列と一致しているか確認してください。")
            # 代替案: 固定インデックスを使う (例: target_domain_index = 0)
            target_domain_index = 0
            print(f"Warning: フォールバックとしてインデックス {target_domain_index} を使用します。")


        ref_img_path = None
        ref_folder_path = os.path.join(args.ref_dir, target_domain_name)
        if os.path.isdir(ref_folder_path):
            ref_img_files = [f for f in os.listdir(ref_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if ref_img_files:
                ref_img_files.sort()
                ref_img_path = os.path.join(ref_folder_path, ref_img_files[0])
        if not ref_img_path:
             print(f"Error: ref_dir のサブフォルダに画像ファイルが見つかりません: {ref_folder_path}")
             return

        print(f"参照画像をロード中: {ref_img_path} (ドメイン: {target_domain_name}, インデックス: {target_domain_index})")
        ref_img_pil = Image.open(ref_img_path).convert('RGB')
        ref_img_pil_resized = preprocess_pil(ref_img_pil) # リサイズのみ適用

        # ドメインインデックスをNumPy配列として準備
        # Core MLの入力仕様 (Int64TensorSpec または Int32TensorSpec) に合わせる
        # ct.convert時にint64で指定してもint32にキャストされることがあるため、モデルのinput_descriptionを確認
        domain_index_np = np.array([target_domain_index], dtype=np.int32) # int32 がより一般的かも
        print(f"参照ドメインインデックス (NumPy): {domain_index_np}, dtype: {domain_index_np.dtype}")

        # --- 推論の実行 ---
        # 入力名は mlmodel.input_description で表示されるものと一致させる必要あり
        # デフォルト: "source_image", "reference_image", "reference_domain_index"
        # ImageType入力にはPIL Imageオブジェクトを渡すのが一般的
        input_dict = {
            "source_image": src_img_pil_resized,
            "reference_image": ref_img_pil_resized,
            "reference_domain_index": domain_index_np
        }

        print("Core ML 推論を実行中...")
        predictions = mlmodel.predict(input_dict)
        print("Core ML 推論完了。")

        # 出力の処理と保存
        # 出力名は mlmodel.output_description で表示されるものと一致させる必要あり
        # デフォルト: "generated_image_tensor"
        output_key = "generated_image_tensor" # モデルディスクリプションで要確認
        output_filename = os.path.join(args.result_dir, f"coreml_output_{target_domain_name}.jpg")
        save_coreml_output(predictions, output_key, output_filename)

        # 比較のためにリサイズされた入力画像も保存
        src_img_pil_resized.save(os.path.join(args.result_dir, "coreml_input_source_resized.jpg"))
        ref_img_pil_resized.save(os.path.join(args.result_dir, "coreml_input_reference_resized.jpg"))
        print("比較用にリサイズされた入力画像を保存しました。")

    except Exception as e:
        print(f"\nError: Core ML 推論テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

@torch.no_grad()
def translate_single_image(nets, args, x_src, x_ref, y_ref, filename):
    """単一の入力画像と参照画像から1枚の変換画像を生成する"""
    # 入力画像のサイズを取得
    _, C, H, W = x_src.size()
    
    # マスクの生成（必要な場合）
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    
    # リファレンス画像からスタイルを抽出
    s_ref = nets.style_encoder(x_ref, y_ref)
    
    # 生成器を使用して画像を変換
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    
    # 変換された画像を保存
    save_image(x_fake, 1, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert StarGAN v2 PyTorch model to Core ML")

    # --- 引数設定 ---
    # Model configuration.
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=23, help='Number of domains in the model/dataset')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension (usually fixed for pretrained model)')
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension (usually fixed for pretrained model)')
    parser.add_argument('--w_hpf', type=float, default=0, help='Weight for high-pass filtering. MUST be 0 for this conversion script.')

    # Paths and directories.
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing PyTorch network checkpoints (_nets_ema.ckpt)')
    parser.add_argument('--resume_iter', type=int, required=True, help='Iteration number of the checkpoint to load (e.g., 100000)')
    parser.add_argument('--result_dir', type=str, default='results_coreml', help='Directory for saving the CoreML model and PyTorch samples')
    # Optional: Directories for PyTorch sampling
    parser.add_argument('--src_dir', type=str, default=None, help='Directory containing source images for PyTorch sampling (optional)')
    parser.add_argument('--ref_dir', type=str, default=None, help='Directory containing reference images for PyTorch sampling (optional)')
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt', help='Path for pretrained FAN (only needed if w_hpf > 0, which is not supported)')

    # Sampling/Loader configuration (only used if src_dir and ref_dir are provided)
    parser.add_argument('--val_batch_size', type=int, default=4, help='Batch size for PyTorch sampling')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for PyTorch sampling DataLoader')
    parser.add_argument('--seed', type=int, default=777, help='Seed for random number generator (affects sampling if shuffle=True)')


    args = parser.parse_args()

    # --- 引数チェック ---
    if args.w_hpf != 0:
        print("Error: This script only supports conversion for models trained with w_hpf=0 (FAN disabled).")
        exit(1)
    if args.src_dir and not args.ref_dir:
        print("Warning: --src_dir provided but --ref_dir is missing. PyTorch sampling will be skipped.")
        args.src_dir = None # Samplingしない
    if not args.src_dir and args.ref_dir:
        print("Warning: --ref_dir provided but --src_dir is missing. PyTorch sampling will be skipped.")
        args.ref_dir = None # Samplingしない

    # Configオブジェクトを作成
    config = Config.from_args(args)

    # main関数を実行
    main(config)
    print("\nScript finished.")