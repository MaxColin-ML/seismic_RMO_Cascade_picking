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

## Citation

If you use this code in your research, please cite:

H. Wang et al., "A Label-Free High-Precision Residual Moveout Picking Method for Depth-Domain Tomography Based on Deep Learning," IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025, Art no. 5922515. doi: 10.1109/TGRS.2025.3612370

BibTeX:
```bibtex
@article{Wang2025,
  author = {H. Wang and others},
  title = {A Label-Free High-Precision Residual Moveout Picking Method for Depth-Domain Tomography Based on Deep Learning},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume = {63},
  pages = {1--15},
  year = {2025},
  note = {Art. no. 5922515},
  doi = {10.1109/TGRS.2025.3612370},
}
```
