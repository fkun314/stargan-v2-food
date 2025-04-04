import coremltools as ct
import numpy as np
from PIL import Image
import argparse
import os
import torch # 非正規化のため（なくても実装可）
import torchvision.utils as vutils # 画像保存のため
from torchvision import transforms # 画像リサイズのため
from typing import Dict

# --- ヘルパー関数: Core ML 出力を画像として保存 ---
def save_coreml_output(output_dict: Dict[str, np.ndarray], output_key: str, filename: str):
    """Core MLモデルのテンソル出力を画像として保存する"""
    if output_key not in output_dict:
        print(f"Error: 出力キー '{output_key}' が推論結果に見つかりません。")
        print(f"利用可能なキー: {output_dict.keys()}")
        return

    output_tensor_np = output_dict[output_key]
    print(f"Core ML Output shape: {output_tensor_np.shape}, dtype: {output_tensor_np.dtype}, min: {output_tensor_np.min()}, max: {output_tensor_np.max()}")

    output_tensor = torch.from_numpy(output_tensor_np)

    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        out = (tensor + 1.0) / 2.0
        return out.clamp_(0.0, 1.0)

    if output_tensor.dim() == 4 and output_tensor.shape[0] == 1:
        denormalized_tensor = denormalize(output_tensor[0])
    else:
        print(f"Warning: 予期しない出力テンソルの形状 {output_tensor.shape}。最初の要素を使用します。")
        denormalized_tensor = denormalize(output_tensor[0] if output_tensor.dim() > 2 else output_tensor)

    try:
        vutils.save_image(denormalized_tensor, filename)
        print(f"Core ML 出力画像を保存しました: {filename}")
    except Exception as e:
        print(f"Error: Core ML 出力画像の保存中にエラーが発生しました: {e}")

# --- メインの推論実行関数 ---
def main(args):
    print("--- Core ML 推論プログラム開始 ---")
    print(f"モデルファイル: {args.model_path}")
    print(f"ソース画像: {args.source_image}")
    print(f"参照画像: {args.reference_image}")
    print(f"参照ドメインインデックス: {args.domain_index}")
    print(f"出力先ディレクトリ: {args.output_dir}")

    # --- 1. モデルのロード ---
    if not os.path.exists(args.model_path):
        print(f"Error: モデルファイルが見つかりません: {args.model_path}")
        return
    try:
        print("Core ML モデルをロード中...")
        mlmodel = ct.models.MLModel(args.model_path)
        print("Core ML モデルのロード成功。")
        print("モデル入力情報:", mlmodel.input_description)
        print("モデル出力情報:", mlmodel.output_description)
    except Exception as e:
        print(f"\nError: モデルのロード中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. 入力データの準備 ---
    try:
        print("入力データを準備中...")
        # 画像サイズを取得 (モデルの入力情報から取得する方が確実)
        try:
             # 入力情報から形状を取得 (例: 'source_image')
             input_desc = mlmodel.input_description['source_image']
             # ct.ImageType の場合 shape プロパティから (C, H, W) または (H, W, C) などを取得
             # ct.TensorType の場合 shape プロパティからタプルで取得
             # ここでは ImageType と仮定し、Channel First (C, H, W) を想定
             input_shape_str = str(input_desc.type.shape) # 文字列から解析する必要がある場合も
             # 簡単のため、固定サイズを使用 (必要なら上記から動的に取得)
             img_size = 256 # モデルに合わせて変更
             print(f"想定される入力画像サイズ: {img_size}x{img_size}")
        except Exception as e:
             print(f"Warning: モデル情報からの画像サイズ取得に失敗 ({e})。デフォルトの256を使用します。")
             img_size = 256

        # リサイズ変換 (モデル訓練時と同じ方法 - ここでは潰しリサイズを仮定)
        preprocess_pil = transforms.Compose([
            transforms.Resize([img_size, img_size], interpolation=transforms.InterpolationMode.BILINEAR),
        ])

        # ソース画像のロードとリサイズ
        if not os.path.exists(args.source_image):
            print(f"Error: ソース画像ファイルが見つかりません: {args.source_image}")
            return
        src_img_pil = Image.open(args.source_image).convert('RGB')
        src_img_pil_resized = preprocess_pil(src_img_pil)

        # 参照画像のロードとリサイズ
        if not os.path.exists(args.reference_image):
            print(f"Error: 参照画像ファイルが見つかりません: {args.reference_image}")
            return
        ref_img_pil = Image.open(args.reference_image).convert('RGB')
        ref_img_pil_resized = preprocess_pil(ref_img_pil)

        # ドメインインデックス (NumPy配列)
        # モデルの入力仕様に合わせる (int32が多い)
        domain_index_np = np.array([args.domain_index], dtype=np.int32)

        print("入力データの準備完了。")
    except Exception as e:
        print(f"\nError: 入力データの準備中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. 推論の実行 ---
    try:
        # デバッグ出力: input_description の型と内容を確認
        print(f"mlmodel.input_description の型: {type(mlmodel.input_description)}")
        print(f"mlmodel.input_description の内容: {mlmodel.input_description}")
        print(f"mlmodel.output_description の型: {type(mlmodel.output_description)}")
        print(f"mlmodel.output_description の内容: {mlmodel.output_description}")

        # 修正: input_description が名前のリスト/タプルであると仮定
        # list() でコピーしてリストとして扱う
        try:
             input_names = list(mlmodel.input_description)
             output_names = list(mlmodel.output_description)
        except TypeError:
             # もし list() でエラーが出る場合（非イテラブルなオブジェクト）
             print("Error: mlmodel.input_description または output_description をリストに変換できません。")
             print("モデルのメタデータ構造が予期しない形式です。coremltools のドキュメントを確認してください。")
             # デバッグ用に属性を表示してみる
             # print(f"input_description の属性: {dir(mlmodel.input_description)}")
             return

        # 取得したリストが文字列のリストであることを確認 (念のため)
        if not all(isinstance(name, str) for name in input_names):
             print(f"Error: 取得した入力名リストに文字列でない要素が含まれています: {input_names}")
             return
        if not all(isinstance(name, str) for name in output_names):
             print(f"Error: 取得した出力名リストに文字列でない要素が含まれています: {output_names}")
             return

        # (オプション) 取得した名前を確認
        print(f"取得された入力名: {input_names}")
        print(f"取得された出力名: {output_names}")

        # 入力名が期待通りか確認 (最低3つ必要)
        if not input_names or len(input_names) < 3:
            print(f"Error: 予期しない入力名のリスト: {input_names}。モデル定義を確認してください。")
            return
        # 出力名が期待通りか確認 (最低1つ必要)
        if not output_names:
            print(f"Error: 予期しない出力名のリスト: {output_names}。モデル定義を確認してください。")
            return

        # 想定される順番 (変換スクリプトで定義した順) でキーを設定
        source_key = input_names[0]
        ref_key = input_names[1]
        index_key = input_names[2]
        output_key = output_names[0]

        print(f"推論に使用する入力キー: {source_key}, {ref_key}, {index_key}")
        print(f"推論に使用する出力キー: {output_key}")

        # input_dict の作成は変更なし
        input_dict = {
            source_key: src_img_pil_resized,
            ref_key: ref_img_pil_resized,
            index_key: domain_index_np
        }
        print(f"推論への入力型: { {k: type(v) for k, v in input_dict.items()} }")
        if index_key in input_dict and isinstance(input_dict[index_key], np.ndarray):
            print(f"{index_key} dtype: {input_dict[index_key].dtype}")
        else:
             print(f"Warning: {index_key} が input_dict に存在しないか、NumPy 配列ではありません。")

        print("Core ML 推論を実行中...")
        predictions = mlmodel.predict(input_dict)
        print("Core ML 推論完了。")
        print(f"推論結果のキー: {list(predictions.keys())}") # predictions は辞書

    except IndexError as e:
         # input_names[2] などでインデックス範囲外になった場合
         print(f"\nError: インデックスエラーが発生しました: {e}")
         print(f"検出された入力名: {input_names}")
         print(f"検出された出力名: {output_names}")
         print("入力/出力名の数が想定と異なる可能性があります。")
         import traceback
         traceback.print_exc()
         return
    except Exception as e:
        print(f"\nError: Core ML 推論中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. 出力の保存 ---
    try:
        print("推論結果を保存中...")
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = os.path.join(args.output_dir, "inference_output.jpg")

        # 出力キーが正しいか確認 (推論結果のキー表示を参考にする)
        if output_key not in predictions:
             print(f"Error: 期待される出力キー '{output_key}' が推論結果に含まれていません。")
             print(f"利用可能なキー: {predictions.keys()}")
             # 利用可能な最初のキーを使うなどのフォールバック
             if predictions:
                 output_key = list(predictions.keys())[0]
                 print(f"フォールバックとしてキー '{output_key}' を使用します。")
             else:
                 print("Error: 推論結果が空です。")
                 return

        save_coreml_output(predictions, output_key, output_filename)

        # 比較用に入力画像も保存
        src_img_pil_resized.save(os.path.join(args.output_dir, "inference_input_source.jpg"))
        ref_img_pil_resized.save(os.path.join(args.output_dir, "inference_input_reference.jpg"))
        print("比較用にリサイズされた入力画像を保存しました。")

    except Exception as e:
        print(f"\nError: 出力の保存中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    print("--- Core ML 推論プログラム終了 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference using a Core ML StarGAN v2 model.")

    parser.add_argument('--model_path', type=str, required=True, help='Path to the .mlpackage file.')
    parser.add_argument('--source_image', type=str, required=True, help='Path to the source input image.')
    parser.add_argument('--reference_image', type=str, required=True, help='Path to the reference style image.')
    parser.add_argument('--domain_index', type=int, required=True, help='Target domain index for the reference image.')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Directory to save the output image.')

    args = parser.parse_args()
    main(args)