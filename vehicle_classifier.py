"""
車種分類器 - 小型車と大型車を判別するAIモデル
交通量カウンターとの統合用クラス
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path


class VehicleClassifier:
    """車種分類器クラス（小型車 vs 大型車）"""
    
    def __init__(self, model_path="car_classfier/vehicle_model.pt", device=None):
        """
        Args:
            model_path: 学習済みモデルのパス
            device: 使用デバイス（cuda/cpu）
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        self.confidence_threshold = 0.6  # 信頼度閾値
        
        # クラスラベル（アルファベット順）
        self.idx_to_class = {0: 'large', 1: 'small'}
        self.class_to_idx = {'large': 0, 'small': 1}
        
        # 前処理用Transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # モデル初期化
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """学習済みモデルを読み込み"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")

            # EfficientNet-B1ベースのモデル
            self.model = models.efficientnet_b1(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
            
            # 学習済み重みを読み込み
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            return True
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            self.is_loaded = False
            return False
    
    def classify_image(self, image):
        """
        画像を分類
        
        Args:
            image: PIL.Image または numpy.ndarray
            
        Returns:
            dict: {
                'class_name': str,      # 'large', 'small', 'unknown'
                'confidence': float,    # 信頼度 [0-1]
                'probabilities': dict   # {'large': prob, 'small': prob}
            }
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    'class_name': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {'large': 0.0, 'small': 0.0}
                }
        
        try:
            # PIL.Imageに変換
            if isinstance(image, np.ndarray):
                # 独立したコピーを作成（Qtとのメモリ競合を防ぐ）
                image = np.ascontiguousarray(image.copy())
                # OpenCV BGR -> RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = image[:, :, ::-1]  # BGR to RGB
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError("Unsupported image type")
            
            # RGB変換
            image = image.convert('RGB')
            
            # 前処理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推論
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu()
            
            # 結果処理
            confidence, pred_idx = probabilities.max(0)
            class_name = self.idx_to_class[pred_idx.item()]
            confidence_value = confidence.item()
            
            # 閾値チェック
            if confidence_value < self.confidence_threshold:
                class_name = 'unknown'
            
            prob_dict = {
                'large': probabilities[0].item(),
                'small': probabilities[1].item()
            }
            
            return {
                'class_name': class_name,
                'confidence': confidence_value,
                'probabilities': prob_dict
            }
            
        except Exception as e:
            print(f"画像分類エラー: {e}")
            return {
                'class_name': 'unknown',
                'confidence': 0.0,
                'probabilities': {'large': 0.0, 'small': 0.0}
            }
    
    def classify_batch(self, images):
        """
        複数画像をバッチ処理
        
        Args:
            images: list of PIL.Image or numpy.ndarray
            
        Returns:
            list: 分類結果のリスト
        """
        results = []
        for image in images:
            result = self.classify_image(image)
            results.append(result)
        return results
    
    def set_confidence_threshold(self, threshold):
        """信頼度閾値を設定"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def get_model_info(self):
        """モデル情報を取得"""
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'classes': list(self.class_to_idx.keys()),
            'is_loaded': self.is_loaded
        }