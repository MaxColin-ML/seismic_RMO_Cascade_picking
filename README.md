# Code Explanation 

## Data Preparation This study requires two types of data: 

1. Simulated data, used for model training; 

2. Real-world data, parsed based on the Segy format. 

- Simulated Data: The folder 'scripts/a_generate_data' contains the code for generating simulated data. 

- Parsing Real-World Data: The folder 'scripts/b_parse_field_data' contains the code for parsing real-world data. 

- Predicting Other Data: The folder 'scripts/z_predict_new_data' contains the code for predicting new real-world data; however, please parse the data first. 

## Model Training 

In this study, we directly use parameter tuning to find the optimal training hyperparameters. 

- Tuning Paradigm: The folder 'scripts/ep_tuning_parameters' contains the relevant tuning code. 

## Model Testing 

Automatic testing is performed after each training iteration. Refer to the file 'models/cascade_method/train_run.py', which contains the main program code. 

## Trained Model 

The trained model file is saved in 'run_demo/data_model/model_checkpoint.pth'