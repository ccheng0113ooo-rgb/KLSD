import json
import time
import logging
import traceback
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class PredictConfig:
    BASE_MODEL_DIR = r"D:\Desktop\backend\back\server\model_files\tradition_models"
    DEVICE = "cpu"

class TDTPredictor:
    def __init__(self, config):
        self.config = config
        self.targets = ['jak1', 'jak2', 'jak3', 'tyk2']
        self.models = {
            'svm': {},
            'knn': {},
            'xgboost': {},
            'rf': {}
        }
        self._load_models()

    def _load_models(self):
        model_file_mapping = {
            'svm': 'SVM_{}.sav',
            'knn': 'KNN_{}.pkl',
            'xgboost': 'XGBoost_{}.sav',
            'rf': 'RF_{}.sav'
        }

        for target in self.targets:
            target_upper = target.upper()
            for model_type, filename_pattern in model_file_mapping.items():
                try:
                    model_filename = filename_pattern.format(target_upper)
                    model_path = os.path.join(self.config.BASE_MODEL_DIR, model_filename)

                    if not os.path.exists(model_path):
                        logger.error(f"模型文件不存在: {model_path}")
                        continue

                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

                    self.models[model_type][target] = model
                    logger.info(f"成功加载 {model_type.upper()} 模型 for {target}")
                except Exception as e:
                    logger.error(f"加载 {model_type.upper()} 模型 for {target} 失败: {str(e)}")
                    self.models[model_type][target] = None

    def smiles_to_maccs(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            return {f'bit{i}': int(bit) for i, bit in enumerate(maccs)}
        except Exception as e:
            logger.error(f"SMILES转换MACCS失败: {str(e)}")
            return None

    def predict(self, smiles, model_type, target):
        model = self.models[model_type].get(target)
        if model is None:
            return {
                "prediction": "error",
                "probability": 0.0,
                "error": f"{model_type.upper()} model not loaded for {target}"
            }

        try:
            features = self.smiles_to_maccs(smiles)
            if features is None:
                return {
                    "prediction": "error",
                    "probability": 0.0,
                    "error": "无法从SMILES生成MACCS特征"
                }

            features_df = pd.DataFrame([features])

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_df)[0][1]
            else:
                proba = model.decision_function(features_df)[0]
                proba = 1 / (1 + np.exp(-proba))

            prediction = "active" if proba >= 0.5 else "inactive"

            return {
                "prediction": prediction,
                "probability": float(proba),
                "error": None
            }
        except Exception as e:
            return {
                "prediction": "error",
                "probability": 0.0,
                "error": str(e)
            }

    def predict_all_models(self, smiles):
        results = {
            "tdt_predictions": {}
        }

        for target in self.targets:
            target_results = {}
            for model_type in ['svm', 'knn', 'xgboost', 'rf']:
                target_results[model_type] = self.predict(smiles, model_type, target)
            results["tdt_predictions"][target] = target_results

        return results

def main():
    parser = argparse.ArgumentParser(description='传统机器学习模型预测脚本')
    parser.add_argument('--smiles', required=True, help='输入的SMILES字符串')
    args = parser.parse_args()

    try:
        logger.info(f"开始处理SMILES: {args.smiles}")

        config = PredictConfig()
        predictor = TDTPredictor(config)

        results = predictor.predict_all_models(args.smiles)

        response = {
            "data": results,
            "status": "success",
            "timestamp": int(time.time() * 1000)
        }

        print(json.dumps(response, ensure_ascii=False))
        sys.stdout.flush()

    except Exception as e:
        error_msg = {
            "error": str(e),
            "type": "fatal_error",
            "traceback": traceback.format_exc(),
            "timestamp": int(time.time() * 1000)
        }
        print(json.dumps(error_msg, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()