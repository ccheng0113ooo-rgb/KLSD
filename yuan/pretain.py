# # ##只包含smile分子式重建功能
# import argparse
# import os
# import numpy as np
# import torch
# from sklearn.preprocessing import StandardScaler
# from transformer import XFormer
# import logging
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class EarlyStopper:
#     def __init__(self, patience=30, min_delta=0.001):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.min_validation_loss = float('inf')

#     def early_stop(self, validation_loss):
#         if validation_loss < self.min_validation_loss:
#             self.min_validation_loss = validation_loss
#             self.counter = 0
#         elif validation_loss > (self.min_validation_loss + self.min_delta):
#             self.counter += 1
#             if self.counter >= self.patience:
#                 return True
#         return False

# def visualize_embeddings(embeddings, epoch):
#     """简化版可视化函数"""
#     plt.figure(figsize=(10, 8))
#     plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)
#     plt.title(f'Feature Space at Epoch {epoch}')
#     os.makedirs("visualizations", exist_ok=True)
#     plt.savefig(f"visualizations/embeddings_epoch_{epoch}.png")
#     plt.close()

# def xformer_train(**kwargs):
#     # 设备设置
#     device = torch.device(kwargs["device"] if torch.cuda.is_available() else "cpu")
#     torch.backends.cudnn.benchmark = True
#     logger.info(f"Using device: {device}")
    
#     try:
#         # 数据加载
#         x_data = np.load(kwargs["x_data"])
#         y_data = torch.load(kwargs["y_data"], map_location="cpu").numpy()
        
#         assert x_data.shape[0] == len(y_data) == 18086, "数据形状不匹配"
#         logger.info(f"数据加载成功: x.shape={x_data.shape}, y.shape={y_data.shape}")
        
#         # 数据预处理
#         scaler = StandardScaler()
#         x_data = scaler.fit_transform(x_data.reshape(18086, -1)).reshape(-1, 128, 40)
        
#         # 数据划分
#         train_size = int(0.9 * len(x_data))
#         x_train, x_val = x_data[:train_size], x_data[train_size:]
#         y_train, y_val = y_data[:train_size], y_data[train_size:]
        
#         # 创建数据集
#         train_dataset = torch.utils.data.TensorDataset(
#             torch.tensor(x_train, dtype=torch.float32),
#             torch.tensor(y_train, dtype=torch.float32)
#         )
#         val_dataset = torch.utils.data.TensorDataset(
#             torch.tensor(x_val, dtype=torch.float32),
#             torch.tensor(y_val, dtype=torch.float32)
#         )

#         train_loader = torch.utils.data.DataLoader(
#             train_dataset,
#             batch_size=kwargs["batch_size"],
#             shuffle=True,
#             num_workers=4
#         )
#         val_loader = torch.utils.data.DataLoader(
#             val_dataset,
#             batch_size=kwargs["batch_size"],
#             num_workers=4
#         )

#     except Exception as e:
#         logger.error(f"数据加载失败: {str(e)}")
#         raise

#     # 模型初始化
#     model = XFormer(
#         device=device,
#         batch_first=True,
#         vocab="#()+-./123456789=@BCFHIKNOPS[\]acilnors",
#         mask_rate=0.2,
#         lr=kwargs["lr"],
#         tag=kwargs["tag"],
#         # seq_length=128
#     ).to(device)
    
#     # 训练优化
#     early_stopper = EarlyStopper(patience=30)
#     model.add_optimizer()
    
#     try:
#         best_val_loss = float('inf')
#         for epoch in range(kwargs["max_epochs"]):
#             # 训练阶段
#             train_loss = model.train_epoch(train_loader)
            
#             # 验证阶段
#             val_loss, recon_coeff = model.validate(val_loader)
            
#             # 学习率调整
#             model.scheduler.step(val_loss)
            
#             # 早停检查
#             if early_stopper.early_stop(val_loss):
#                 logger.info(f"早停触发于第 {epoch} 轮")
#                 break
                
#             # 保存最佳模型
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 model_path = f"models/{kwargs['tag']}_best.pt"
#                 os.makedirs("models", exist_ok=True)
#                 torch.save(model.state_dict(), model_path)
#                 logger.info(f"最佳模型已保存至: {model_path}")
            
#             # 可视化
#             if epoch % 50 == 0:
#                 with torch.no_grad():
#                     sample_x, _ = next(iter(val_loader))
#                     embeddings = model.get_embeddings(sample_x.to(device)).cpu().numpy()
#                     visualize_embeddings(embeddings, epoch)
            
#             logger.info(
#                 f"Epoch {epoch+1}/{kwargs['max_epochs']} | "
#                 f"Train Loss: {train_loss:.4f} | "
#                 f"Val Loss: {val_loss:.4f} | "
#                 f"Recon Coeff: {recon_coeff:.4f}"
#             )
            
#     except RuntimeError as e:
#         logger.error(f"训练错误: {str(e)}")
#         raise

#     # 最终模型保存
#     final_model_path = f"models/{kwargs['tag']}_final.pt"
#     torch.save(model.state_dict(), final_model_path)
#     logger.info(f"最终模型已保存至: {final_model_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-x", "--x_data", required=True)
#     parser.add_argument("-y", "--y_data", required=True)
#     parser.add_argument("-b", "--batch_size", type=int, default=64)
#     parser.add_argument("-l", "--lr", type=float, default=5e-5)
#     parser.add_argument("-m", "--max_epochs", type=int, default=750)
#     parser.add_argument("-d", "--device", default="cuda")
#     parser.add_argument("-t", "--tag", required=True)
    
#     args = parser.parse_args()
    
#     # 参数验证
#     assert os.path.exists(args.x_data), "特征文件不存在"
#     assert os.path.exists(args.y_data), "目标文件不存在"
#     assert args.batch_size > 0, "batch_size必须大于0"
    
#     # 设置随机种子
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
    
#     xformer_train(**vars(args))



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

    