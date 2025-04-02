import os
import torch
from PIL import Image
from core.model import Generator
from core.model import StyleEncoder
import torchvision.transforms as transforms

# 画像の前処理関数を定義
def preprocess_image(img, img_size=256):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0, 1] の範囲に変換
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] に正規化
    ])
    return transform(img)

# 画像の後処理関数を定義
def postprocess_image(tensor):
    tensor = (tensor + 1) / 2  # [-1, 1] から [0, 1] に戻す
    tensor = tensor.clamp(0, 1)  # 範囲を制限
    tensor = tensor * 255  # [0, 255] にスケール
    tensor = tensor.byte()  # 整数に変換
    img = transforms.ToPILImage()(tensor)
    return img

def generate_samples(checkpoint_path, src_dir, ref_dir, output_dir):
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # チェックポイントからモデルをロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(checkpoint.keys())
    image_size = 256  # 学習時と同じサイズ
    num_domains = 23  # 学習時と同じドメイン数
    style_dim = 64    # デフォルト値（学習時の設定に合わせる）

    generator = Generator(img_size=image_size, style_dim=style_dim).to(device)
    style_encoder = StyleEncoder(img_size=image_size, style_dim=style_dim, num_domains=num_domains).to(device)
    generator.load_state_dict(checkpoint['generator'])
    style_encoder.load_state_dict(checkpoint['style_encoder'])
    generator.eval()
    style_encoder.eval()

    # ソース画像のロード
    source_images = []
    for filename in os.list_dir(src_dir):
        img = Image.open(os.path.join(src_dir, filename)).convert('RGB')
        preprocessed_img = preprocess_image(img)
        source_images.append(preprocessed_img)

    # 参照画像のロード
    reference_images = []
    for filename in os.list_dir(ref_dir):
        img = Image.open(os.path.join(ref_dir, filename)).convert('RGB')
        preprocessed_img = preprocess_image(img)
        reference_images.append(preprocessed_img)

    # サンプル生成
    for src_idx, src_img in enumerate(source_images):
        for ref_idx, ref_img in enumerate(reference_images):
            with torch.no_grad():
                style_code = style_encoder(ref_img.unsqueeze(0))
                translated_img = generator(src_img.unsqueeze(0), style_code)
                translated_img = postprocess_image(translated_img[0])
                output_filename = f"src_{src_idx}_ref_{ref_idx}.png"
                translated_img.savefig(os.path.join(output_dir, output_filename))

if __name__ == "__main__":
    checkpoint_path = 'expr/checkpoints/foodimg512/200000_nets_ema.ckpt'
    src_dir = 'data/foodimg512/val'
    ref_dir = 'data/foodimg512/val'
    output_dir = 'data/foodimg512_output'
    generate_samples(checkpoint_path, src_dir, ref_dir, output_dir)