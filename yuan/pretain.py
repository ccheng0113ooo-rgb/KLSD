

##(1) 新增图卷积模块(2) 特征融合与互信息计算(融合序列特征和图特征\监控Transformer特征与RDKit特征的冗余性)

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 添加缺失的导入
from sklearn.preprocessing import StandardScaler
import logging
from transformer import XFormer
from seq_data_loader import EnhancedSeqDataLoader
from data import make_token_dict  # 添加缺失的导入
import matplotlib.pyplot as plt
import torch.nn.functional as F
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopper:
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def evaluate(model, dataloader, device):
    """评估函数"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += F.mse_loss(output[:, 0], y).item()
    return total_loss / len(dataloader)

def xformer_train(**kwargs):
    # 设备设置
    device = torch.device(kwargs["device"] if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    try:
        # 1. 初始化数据加载器（先获取动态长度）
        loader = EnhancedSeqDataLoader(
            token_dict=make_token_dict(),
            coverage=kwargs.get("coverage", 0.95)
        )
        logger.info(f"动态计算seq_length: {loader.seq_length}")
        
        # 2. 加载数据
        train_data = loader.load_and_align_data(kwargs["train_csv"])
        val_data = loader.load_and_align_data(kwargs.get("val_csv", kwargs["train_csv"]))  # 添加验证集加载
        
        train_features = loader.generate_features(train_data["smiles"])
        val_features = loader.generate_features(val_data["smiles"])
        
        # 3. 数据标准化
        scaler = StandardScaler()
        orig_shape = train_features.shape
        train_features = scaler.fit_transform(train_features.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        val_features = scaler.transform(val_features.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        
        # 4. 创建数据集
        train_dataset = TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_data["labels"], dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(val_features, dtype=torch.float32),
            torch.tensor(val_data["labels"], dtype=torch.float32)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=kwargs["batch_size"],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(  # 定义val_loader
            val_dataset,
            batch_size=kwargs["batch_size"],
            shuffle=False,
            num_workers=4
        )
        
        # 5. 初始化模型
        model = XFormer(
            seq_length=loader.seq_length,
            d_model=kwargs.get("d_model", 256),
            nhead=kwargs.get("nhead", 8),
            num_layers=kwargs.get("num_layers", 6),
            use_graph=kwargs.get("use_graph", True),
            compute_mi=kwargs.get("compute_mi", True)
        ).to(device)
        
        # 6. 训练循环
        optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs["lr"])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stop = EarlyStopper(patience=30)
        
        for epoch in range(kwargs["max_epochs"]):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = F.mse_loss(output[:, 0], y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证和早停
            val_loss = evaluate(model, val_loader, device)
            scheduler.step(val_loss)
            
            if early_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            logger.info(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", help="可选验证集路径")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--use_graph", action="store_true")
    parser.add_argument("--compute_mi", action="store_true")
    args = parser.parse_args()
    
    xformer_train(**vars(args))


    
