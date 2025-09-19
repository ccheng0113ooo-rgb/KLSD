import logging
import numpy as np
import torch

NULL_ELEMENT = 'NULL_ELEMENT'

# 定义有效的 SMILES 字符集
VALID_SMILES_CHARS = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@=+-/\\()[]{}#.')

def make_token_dict() -> dict:
    """生成包含所有必需化学字符的固定字典"""
    token_dict = {NULL_ELEMENT: 0}  # 确保NULL_ELEMENT索引为0
    for idx, char in enumerate(sorted(VALID_SMILES_CHARS), 1):
        token_dict[char] = idx
    return token_dict

# 程序初始化时创建全局字典
GLOBAL_TOKEN_DICT = make_token_dict()
print(f"字典大小: {len(GLOBAL_TOKEN_DICT)}")

def tokens_to_one_hot(tokens, token_dict, pad_to=100):
    """修正后的one-hot编码函数"""
    pad_classes_to = len(token_dict)
    one_hot = torch.zeros((pad_to, pad_classes_to), dtype=torch.float32)
    
    # 确保token索引有效
    tokens = tokens.long().clamp(0, pad_classes_to-1)
    
    # 使用scatter_实现严格one-hot
    one_hot.scatter_(1, tokens.unsqueeze(1), 1.0)
    return one_hot

def sequence_to_vectors(sequence, sequence_dict, pad_to=64):
    """将SMILES序列转换为向量表示"""
    try:
        vectors = []
        for element in sequence:
            if element in sequence_dict:
                vectors.append(sequence_dict[element])
            else:
                vectors.append(sequence_dict[NULL_ELEMENT])
        
        # 转换为numpy数组并填充
        vectors = np.array(vectors, dtype=np.int64)
        if len(vectors) < pad_to:
            padding = np.full((pad_to - len(vectors),), sequence_dict[NULL_ELEMENT], dtype=np.int64)  # 修正后的代码
            vectors = np.concatenate([vectors, padding])
        return torch.tensor(vectors, dtype=torch.long)
    except Exception as e:
        logging.error(f"序列转换错误: {str(e)}")
        return None

def one_hot_to_sequence(one_hot, sequence_dict):
    """将 one-hot 编码的张量转换回 SMILES 字符串"""
    sequence = ""
    key_dict = {v: k for k, v in sequence_dict.items()}
    for element in one_hot:
        index = torch.argmax(element).item()
        if index in key_dict and key_dict[index] != NULL_ELEMENT:
            sequence += key_dict[index]
    return sequence

def safe_sequence_to_vectors(smi, token_dict):
    """安全的序列转换，带有错误处理"""
    try:
        return sequence_to_vectors(smi, token_dict)
    except Exception as e:
        logging.error(f"安全转换失败: {str(e)}")
        return None

# ##原
# import numpy as np 

# import torch

# NULL_ELEMENT = None 

# def make_token_dict(vocabulary: str) -> dict:

#     token_dict = {}

#     vocabulary_list = list(vocabulary)
#     vocabulary_list.sort()

#     token_dict[NULL_ELEMENT] = np.array(0).reshape(-1,1)
#     for token, element in enumerate(vocabulary):
#         token_dict[element] = torch.tensor(np.array(token + 1)).long().reshape(1,-1)

#     return token_dict

# def tokens_to_one_hot(tokens, pad_to=100, pad_classes_to=33):

#     one_hot = torch.zeros(*tokens.shape[:-1], pad_classes_to)

#     for ii in range(tokens.shape[0]):
#         one_hot[ii, tokens[ii,:].long().item()] = 1.0

#     for jj in range(ii, pad_to):

#         one_hot[jj, 0] = 1.0

#     return one_hot 

# def sequence_to_vectors(sequence, sequence_dict, pad_to=64):

#     vectors = None

#     for element in sequence:

#         if vectors is None:
#             vectors = sequence_dict[element]
#         else:
#             vectors = np.append(vectors, sequence_dict[element], axis=0)

#     while vectors.shape[0] < pad_to:
#         vectors = np.append(vectors, sequence_dict[NULL_ELEMENT], axis=0)

#     return torch.tensor(vectors, dtype=torch.float32)

# def one_hot_to_sequence(one_hot, sequence_dict):

#     sequence = ""

#     key_dict = {sequence_dict[key].item(): key for key in sequence_dict.keys()}
#     for element in one_hot:

#         index = torch.argmax(element).long().item()
#         if index in key_dict.keys():
#             sequence += key_dict[index] if index != 0 else ""
#         else:
#             print(index, "key not in dict")
#             sequence += "" #key_dict[list(key_dict.keys())[0]]

#     return sequence