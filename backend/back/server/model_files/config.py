import torch

class PredictConfig:
    SAVE_DIR = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\optimized_jak_results_finaldata_roc"
    COMPARE_MODEL_DIR = r"D:\CC\PycharmProjects\GoGT\JAK_ML\new_model"
    ACTIVITY_MODEL_DIR = r"D:\desktop\JAKInhibition-master\JAKInhibition-master\nn_results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 1024
    ATOM_FEATURES = ['atomic_num', 'degree', 'formal_charge', 'hybridization',
                    'aromatic', 'num_hs', 'chirality']