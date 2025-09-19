
# KLSD
In this project, we propose a dual-objective predictive model that integrates molecular feature engineering with deep learning to simultaneously predict both the activity and selectivity of inhibitors targeting the JAK kinase family (JAK1, JAK2, JAK3, and TYK2). The model employs a multilayer perceptron (MLP) residual network architecture with target-specific branching to enable accurate prediction across multiple sub-targets. Moreover, it introduces an innovative approach by using quantitative activity values in place of conventional binary labels.

# Requirements
The file is required to be located in requirements.txt.

# Data
finaldata: the input dataset used as the model input
valid_activity_jak.CSV：Raw data of the JAK kinase family
processed: Processed dataset used as input to the model
optimized_jak_results_finaldata_roc1、nn_results_finalall_6：Results of the dual-objective predictive model

# Code
process_data.py、split_dataset.py:Preprocessing and splitting of the input dataset
nn.py: This function contains the network framework of our entire model.

# Train and test folds
python process_data.py --finaldata /Your path
finaldata: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)
python split_dataset.py
python nn.py --finaldata/process /Your path
All files of Code should be stored in the same folder to run the model.

# Contact
If you have any questions or suggestions with the code, please let us know. Contact chen at ccheng0113ooo@gmail.com
