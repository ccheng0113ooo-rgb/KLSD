# # ##只包含smile分子式重建功能
# # # #######加入对transformer模型的评估
# import os
# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt
# from data import make_token_dict

# class XFormer(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
        
#         # 设备初始化
#         self.device = torch.device(kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
#         # 模型参数
#         self.batch_first = kwargs.get("batch_first", True)
#         self.vocab = kwargs.get("vocab", "#()+-./123456789=@BCFHIKNOPS[\]acilnors")
#         self.token_dict = make_token_dict(self.vocab)
#         self.d_model = kwargs.get("d_model", len(self.vocab))
#         self.nhead = kwargs.get("nhead", 8)
#         self.num_layers = kwargs.get("num_layers", 6)
#         self.dropout = kwargs.get("dropout", 0.1)
        
#         # 调整维度确保能被nhead整除
#         if self.d_model % self.nhead != 0:
#             self.d_model = ((self.d_model // self.nhead) + 1) * self.nhead
        
#         # Transformer 层
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model,
#             nhead=self.nhead,
#             dim_feedforward=2048,
#             dropout=self.dropout,
#             batch_first=self.batch_first,
#             device=self.device
#         )
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=self.d_model,
#             nhead=self.nhead,
#             dim_feedforward=2048,
#             dropout=self.dropout,
#             batch_first=self.batch_first,
#             device=self.device
#         )
        
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        
#         # 训练参数
#         self.mask_rate = kwargs.get("mask_rate", 0.2)
#         self.lr = kwargs.get("lr", 5e-5)
#         self.tag = kwargs.get("tag", "transformer")
#         self.seq_length = kwargs.get("seq_length", 128)
#         self.optimizer = None
#         self.scheduler = None

#         # 初始化权重
#         self._init_weights()

#     def _init_weights(self):
#         """初始化权重"""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def add_optimizer(self, lr=None):
#         """初始化优化器和学习率调度器"""
#         if lr is not None:
#             self.lr = lr
#         self.optimizer = torch.optim.AdamW(
#             self.parameters(), 
#             lr=self.lr, 
#             weight_decay=1e-5
#         )
#         self.scheduler = ReduceLROnPlateau(
#             self.optimizer, 
#             mode='min',
#             factor=0.5,
#             patience=5,
#             # verbose=True
#         )

#     def forward(self, x):
#         """带设备检查的前向传播"""
#         if not isinstance(x, torch.Tensor):
#             x = torch.tensor(x, device=self.device)
#         elif x.device != self.device:
#             x = x.to(self.device)
            
#         # 编码器-解码器结构
#         memory = self.encoder(x)
#         output = self.decoder(x, memory)
#         return output
#     def train_epoch(self, dataloader):
#         """训练一个epoch"""
#         self.train()
#         total_loss = 0.0
#         # 修正GradScaler初始化（兼容新旧版本PyTorch）
#         scaler = None
#         if 'cuda' in str(self.device):
#             try:
#             # 新版本API
#                 scaler = torch.amp.GradScaler(device_type='cuda')
#             except TypeError:
#             # 旧版本API
#                 scaler = torch.cuda.amp.GradScaler()
    
#         for batch in dataloader:
#             inputs = batch[0].to(self.device)
#             self.optimizer.zero_grad()
        
#         # 自动混合精度
#             with torch.autocast(device_type=self.device.type, dtype=torch.float16):
#                 outputs = self.forward(inputs)
#                 loss = F.mse_loss(outputs, inputs)
            
#             # L2正则化
#                 l2_reg = torch.tensor(0., device=self.device)
#                 for param in self.parameters():
#                     l2_reg += torch.norm(param)
#                 loss += 1e-5 * l2_reg

#         # 梯度缩放和更新
#             if scaler:
#                 scaler.scale(loss).backward()
#                 scaler.step(self.optimizer)
#                 scaler.update()
#             else:
#                 loss.backward()
#                 self.optimizer.step()
            
#             total_loss += loss.item()
        
#         return total_loss / len(dataloader)
#     def validate(self, dataloader):
#         """验证步骤"""
#         self.eval()
#         total_loss = 0.0
#         total_reconstruction = 0.0
#         total_variance = 0.0
        
#         with torch.no_grad():
#             for batch in dataloader:
#                 inputs = batch[0].to(self.device)
#                 outputs = self.forward(inputs)
                
#                 # 计算损失
#                 loss = F.mse_loss(outputs, inputs)
#                 total_loss += loss.item()
                
#                 # 计算重建系数
#                 residuals = inputs - outputs
#                 mean_input = inputs.mean(dim=0, keepdim=True)
                
#                 total_reconstruction += (residuals ** 2).sum().item()
#                 total_variance += ((inputs - mean_input) ** 2).sum().item()
        
#         avg_loss = total_loss / len(dataloader)
#         recon_coeff = 1 - (total_reconstruction / total_variance)
#         return avg_loss, recon_coeff

#     def get_embeddings(self, x):
#         """获取编码器输出作为特征"""
#         with torch.no_grad():
#             if not isinstance(x, torch.Tensor):
#                 x = torch.tensor(x, device=self.device)
#             elif x.device != self.device:
#                 x = x.to(self.device)
#             return self.encoder(x)

#     def fit(self, train_loader, val_loader=None, max_epochs=100, verbose=True):
#         """完整的训练流程"""
#         self.add_optimizer()
        
#         train_losses = []
#         val_losses = []
#         recon_coeffs = []
#         best_loss = float('inf')

#         for epoch in range(max_epochs):
#             # 训练阶段
#             train_loss = self.train_epoch(train_loader)
#             train_losses.append(train_loss)
            
#             # 验证阶段
#             if val_loader:
#                 val_loss, recon_coeff = self.validate(val_loader)
#                 val_losses.append(val_loss)
#                 recon_coeffs.append(recon_coeff)
                
#                 # 学习率调整
#                 self.scheduler.step(val_loss)
                
#                 # 保存最佳模型
#                 if val_loss < best_loss:
#                     best_loss = val_loss
#                     torch.save(self.state_dict(), f"best_model_{self.tag}.pt")
            
#             # 日志打印
#             if verbose and (epoch % max(1, max_epochs//10) == 0 or epoch == max_epochs-1):
#                 log_msg = f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f}"
#                 if val_loader:
#                     log_msg += f" | Val Loss: {val_loss:.4f} | Recon Coeff: {recon_coeff:.4f}"
#                 print(log_msg)
        
#         # 绘制训练曲线
#         self._plot_metrics(train_losses, val_losses, recon_coeffs)
#         return self

#     def _plot_metrics(self, train_losses, val_losses, recon_coeffs):
#         """绘制训练指标"""
#         plt.figure(figsize=(15, 5))
        
#         # 损失曲线
#         plt.subplot(1, 2, 1)
#         plt.plot(train_losses, label='Train Loss')
#         if val_losses:
#             plt.plot(val_losses, label='Val Loss')
#         plt.title('Training Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         # 重建系数
#         plt.subplot(1, 2, 2)
#         if recon_coeffs:
#             plt.plot(recon_coeffs, label='Recon Coeff', color='green')
#             plt.title('Reconstruction Coefficient')
#             plt.xlabel('Epoch')
#             plt.ylabel('Coefficient')
#             plt.ylim(-0.1, 1.1)
#             plt.legend()
        
#         plt.tight_layout()
#         plt.savefig(f"training_metrics_{self.tag}.png")
#         plt.close()

#     def save(self, path):
#         """保存完整模型"""
#         torch.save(self, path)
        
#     @classmethod
#     def load(cls, path, device='auto'):
#         """加载完整模型"""
#         if device == 'auto':
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         return torch.load(path, map_location=device)


from collections import Counter
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem
from data import NULL_ELEMENT, make_token_dict
import matplotlib.pyplot as plt
import math


# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SMILESReconstructor(nn.Module):
    def __init__(self, token_dict, seq_length=None, **kwargs):
        super().__init__()
        # 验证token字典
        assert NULL_ELEMENT in token_dict, "NULL_ELEMENT must be in token_dict"
        assert '^' in token_dict, "Start token '^' must be in token_dict"
        assert '$' in token_dict, "End token '$' must be in token_dict"

        self.token_dict = token_dict
        self.idx_to_token = {v: k for k, v in token_dict.items()}
        self.vocab_size = len(token_dict)
        self.seq_length = seq_length

        # 打印关键token验证
        logger.info("\n=== Token字典验证 ===")
        logger.info(f"字典大小: {self.vocab_size}")
        logger.info(f"NULL_ELEMENT索引: {token_dict[NULL_ELEMENT]}")
        logger.info(f"起始符(^)索引: {token_dict['^']}")
        logger.info(f"终止符($)索引: {token_dict['$']}")
        for char in ['C', '=', '[', ']', '@', '1', '(', ')', 'O', 'N']:
            logger.info(f"'{char}': {token_dict.get(char, 'MISSING')}")

        # 模型参数
        self.d_model = kwargs.get('d_model', 512)
        self.nhead = kwargs.get('nhead', 16)
        self.num_layers = kwargs.get('num_layers', 8)
        self.dropout = kwargs.get('dropout', 0.1)

        # 调整维度确保能被nhead整除
        if self.d_model % self.nhead != 0:
            self.d_model = ((self.d_model // self.nhead) + 1) * self.nhead

        # 编码器 - 解码器结构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
            dropout=self.dropout,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
            dropout=self.dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # 嵌入层和输出层
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.output = nn.Linear(self.d_model, self.vocab_size)

        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)

        # 初始化
        self._init_weights()
        self.device = torch.device(kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.to(self.device)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        """改进的前向传播，包含位置编码"""
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        memory = self.encoder(src_emb)
        output = self.decoder(src_emb, memory)
        return self.output(output)

    def reconstruct(self, x, max_len=100):
        """改进的自回归重建方法"""
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            outputs = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

            # 初始输入（起始符）
            inputs = torch.full((batch_size, 1), self.token_dict['^'],
                                dtype=torch.long, device=self.device)

            for i in range(max_len):
                # 预测下一个字符
                pred = self.forward(inputs)
                next_token = pred[:, -1].argmax(dim=-1)  # 获取最后一个位置的预测

                # 正确更新输出张量
                outputs[:, i] = next_token

                # 遇到终止符则停止
                if (next_token == self.token_dict['$']).all():
                    break

                # 更新输入 - 确保保持正确的形状 [batch_size, seq_len]
                inputs = torch.cat([inputs, next_token.unsqueeze(1)], dim=1)

            # 解码结果
            results = []
            for seq in outputs.cpu().numpy():
                tokens = []
                for t in seq:
                    if t == self.token_dict['$']:
                        break
                    if t != self.token_dict[NULL_ELEMENT]:  # 跳过填充
                        tokens.append(t)
                smi = ''.join([self.idx_to_token.get(t, '?') for t in tokens])
                results.append(smi)

            return results


class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
                # 修改为支持 batch_first=True 的格式
        x = x + self.pe[:x.size(1)].transpose(0, 1)  # 从 [max_len, 1, d_model] 转换为 [1, seq_len, d_model]
        return self.dropout(x)


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = AdamW(model.parameters(), lr=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.epochs = kwargs.get('epochs', 200)
        self.device = model.device
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            inputs, mask = batch
            inputs = inputs.to(self.device)
            mask = mask.to(self.device)

            # 目标序列是向右移一位的输入序列
            targets = inputs[:, 1:].contiguous()  # 去掉第一个token并确保连续
            inputs = inputs[:, :-1].contiguous()  # 去掉最后一个token并确保连续
            mask = mask[:, 1:].contiguous()  # 对应的mask

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # 只计算非填充位置的损失
            loss = F.cross_entropy(
                outputs.reshape(-1, self.model.vocab_size),
                targets.view(-1),
                ignore_index=self.model.token_dict[NULL_ELEMENT]  # 忽略填充token
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        if not self.val_loader:
            return float('inf'), 0, 0

        self.model.eval()
        total_loss, correct_chars, total_chars = 0, 0, 0
        valid_smiles = 0
        token_counts = Counter()

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, mask = batch
                inputs = inputs.to(self.device)
                mask = mask.to(self.device)

                targets = inputs[:, 1:].contiguous()
                inputs = inputs[:, :-1].contiguous()
                mask = mask[:, 1:].contiguous()

                outputs = self.model(inputs)
                loss = F.cross_entropy(
                    outputs.reshape(-1, self.model.vocab_size),
                    targets.view(-1),
                    ignore_index=self.model.token_dict[NULL_ELEMENT]
                )
                total_loss += loss.item()

                # 计算准确率时忽略填充
                preds = outputs.argmax(dim=-1)
                non_pad = (targets != self.model.token_dict[NULL_ELEMENT])
                correct_chars += (preds[non_pad] == targets[non_pad]).sum().item()
                total_chars += non_pad.sum().item()

                # 记录token分布
                for seq in preds.cpu().numpy():
                    token_counts.update(seq)

                # 解码并验证SMILES
                try:
                    reconstructions = self.model.reconstruct(inputs)
                    for smi in reconstructions:
                        valid_smiles += int(Chem.MolFromSmiles(smi) is not None)
                except Exception as e:
                    logger.error(f"重建失败: {str(e)}")
                    continue

        avg_loss = total_loss / len(self.val_loader)
        char_acc = correct_chars / max(1, total_chars)
        valid_rate = valid_smiles / max(1, len(self.val_loader.dataset))

        # 打印token分布
        logger.info("\n=== Token分布诊断 ===")
        for idx, count in token_counts.most_common(10):
            char = self.model.idx_to_token.get(idx, '?')
            logger.info(f"Token {idx:3d}: {count:6d}次 | 字符: '{char}'")

        return avg_loss, char_acc, valid_rate

    def fit(self):
        history = {
            'train_loss': [],
            'val_loss': [],
            'char_acc': [],
            'valid_rate': []
        }

        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)

            # 验证
            val_loss, char_acc, valid_rate = self.validate()
            history['val_loss'].append(val_loss)
            history['char_acc'].append(char_acc)
            history['valid_rate'].append(valid_rate)

            # 学习率调整
            self.scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
                logger.info("发现新最佳模型，已保存")

            # 日志输出
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Char Acc: {char_acc:.2%} | "
                f"Valid Rate: {valid_rate:.2%}"
            )

            # 每10个epoch展示示例
            if epoch % 10 == 0:
                self._show_examples()

        # 绘制训练曲线
        self._plot_history(history)
        return history

    def _show_examples(self, n=3):
        """展示重建示例"""
        try:
            sample = next(iter(self.val_loader))
            inputs = sample[0][:n].to(self.device)

            original_smiles = []
            for seq in inputs.cpu().numpy():
                tokens = [t for t in seq if t != self.model.token_dict[NULL_ELEMENT]]
                smi = ''.join([self.model.idx_to_token.get(t, '?') for t in tokens])
                original_smiles.append(smi)

            reconstructed = self.model.reconstruct(inputs)

            logger.info("\n=== 重建示例 ===")
            for orig, recon in zip(original_smiles, reconstructed):
                logger.info(f"原始: {orig}")
                logger.info(f"重建: {recon}")
                logger.info(f"有效: {Chem.MolFromSmiles(recon) is not None}")
                logger.info("-" * 50)
        except Exception as e:
            logger.error(f"展示示例失败: {str(e)}")

    def _plot_history(self, history):
        """绘制训练指标"""
        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.legend()

        # 字符准确率
        plt.subplot(1, 3, 2)
        plt.plot(history['char_acc'], label='Accuracy', color='green')
        plt.title('Character Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)

        # 化学有效性
        plt.subplot(1, 3, 3)
        plt.plot(history['valid_rate'], label='Validity', color='red')
        plt.title('Chemical Validity')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
        logger.info("训练指标图已保存为 training_metrics.png")


def load_datasets(feature_path, token_dict, label_path=None, batch_size=64):
    """加载预处理好的数据，正确处理填充token"""
    features = torch.from_numpy(np.load(feature_path)).float()

    # 将one - hot转换回token索引
    token_indices = features.argmax(dim=-1)

    # 创建掩码标识哪些是真实token (非填充)
    pad_token = token_dict[NULL_ELEMENT]
    mask = (token_indices != pad_token).float()

    valid_indices = []
    idx_to_token = {v: k for k, v in token_dict.items()}
    for i in range(token_indices.shape[0]):
        seq = token_indices[i]
        smi = ''.join([idx_to_token.get(t.item(), '?') for t in seq if t.item() != pad_token])
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_indices.append(i)
        except Exception as e:
            logger.warning(f"过滤无效SMILES: {smi}, 错误信息: {str(e)}")

    if len(valid_indices) == 0:
        logger.error(f"数据文件 {feature_path} 中没有有效的SMILES数据，请检查数据。")
        raise ValueError("没有有效的SMILES数据，请检查数据。")

    token_indices = token_indices[valid_indices]
    mask = mask[valid_indices]

    if label_path:  # 如果有标签（可用于监督微调）
        labels = torch.load(label_path)['labels']
        labels = labels[valid_indices]
        dataset = TensorDataset(token_indices.contiguous(), mask.contiguous(), labels.contiguous())
    else:  # 无监督学习
        dataset = TensorDataset(token_indices.contiguous(), mask.contiguous())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return loader


def main():
        # 先验证RDKit基础功能
    assert Chem.MolFromSmiles("CCO") is not None, "RDKit初始化失败！"
    # 确保token字典包含必要的特殊字符
    token_dict = make_token_dict()
    if NULL_ELEMENT not in token_dict:
        token_dict[NULL_ELEMENT] = 0
    if '^' not in token_dict:
        token_dict['^'] = len(token_dict)
    if '$' not in token_dict:
        token_dict['$'] = len(token_dict)

    # 参数配置
    args = {
        'train_feats': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\cleaned_data\processed\transformer_io\train\features.npy',
        'val_feats': r'D:\desktop\JAKInhibition-master\JAKInhibition-master\cleaned_data\processed\transformer_io\test\features.npy',
        'token_dict': token_dict,
        'batch_size': 64,
        'epochs': 200,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 动态获取序列长度
    train_feats = np.load(args['train_feats'])
    seq_length = train_feats.shape[1]
    args['seq_length'] = seq_length
    logger.info(f"自动检测到序列长度: {seq_length}")

    # 数据加载
    try:
        train_loader = load_datasets(args['train_feats'], args['token_dict'], batch_size=args['batch_size'])
        val_loader = load_datasets(args['val_feats'], args['token_dict'], batch_size=args['batch_size'])
    except ValueError as e:
        logger.error(str(e))
        return

    # 模型初始化
    model = SMILESReconstructor(
        token_dict=args['token_dict'],
        seq_length=seq_length,
        d_model=512,
        device=args['device']
    )

    # 训练
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        lr=1e-5,
        epochs=args['epochs']
    )
    history = trainer.fit()

    # 保存最终模型
    torch.save(model.state_dict(), 'final_model.pt')
    logger.info("训练完成，模型已保存")


if __name__ == "__main__":
    main()