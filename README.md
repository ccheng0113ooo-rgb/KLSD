
The KLSD database, available at http://ai.njucm.edu.cn:8080, is maintained by Nanjing University of Chinese Medicine under a time-based access policy to ensure data integrity, confidentiality, and availability. Access is permitted daily from 06:30 to 23:30 Beijing Time; external access is restricted outside these hours. KLSD is guaranteed to remain freely available for at least two years and supports major browsers on Windows and macOS.
# KLSD
Herein, we present a database, KLSD, which is a curated resource of 787 213 small-molecule kinase inhibitors annotated with 1.8 M quantitative activity records across 428 human kinases, emphasizing selectivity and polypharmacology. Moreover, we introduce a dual-task ensemble that simultaneously regresses pAct and computes selectivity scores. The core is a multi-branch residual multilayer perceptron (MLP) whose branches are kinase-specific; this is augmented by SVM, RF, XGBoost, CNN, GCN, GAT, RGCN and VAE-enhanced graph nets. Continuous potency labels replace categorical classes to improve resolution. Benchmarked on the JAK family (JAK1/2/3, TYK2), the ensemble yields prediction accuracies of ≥ 0.84 for each kinase and 0.98 overall, demonstrating strong generalizability. KLSD and models are freely available at http://ai.njucm.edu.cn:8080.

# Requirements
The file is required to be located in requirements.txt.

# Data
Data Sources: Small-molecule kinase inhibitor datasets, including the JAK family benchmark data, were obtained from ChEMBL (https://www.ebi.ac.uk/chembl/). All processed JAK family data are available on GitHub, with the specific dataset download URLs accessible at: https://github.com/ccheng0113ooo-rgb/KLSD/tree/main/finaldata.
finaldata: the input dataset used as the model input
valid_activity_jak.CSV：Raw data of the JAK kinase family
processed: Processed dataset used as input to the model
optimized_jak_results_finaldata_roc1、nn_results_finalall_6：Results of the dual-objective predictive model

# Code
process_data.py、split_dataset.py:Preprocessing and splitting of the input dataset
nn.py: This function contains the network framework of our entire model.
ML_code:Relevant code for the baseline model
cnn:Relevant code for the baseline model
backend:Backend files of the database platform
klsd:Frontend files of the database platform

# Train and test folds
python process_data.py --finaldata /Your path
finaldata: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)
python split_dataset.py
python nn.py --finaldata/process /Your path
All files of Code should be stored in the same folder to run the model.

# Contact
If you have any questions or suggestions with the code, please let us know. Contact chen at ccheng0113ooo@gmail.com
