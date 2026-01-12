# 代码说明

## 数据准备
在本研究中，共需准备两种数据，1模拟数据，用于训练模型，2实际数据，基于segy格式的数据进行解析。

- 模拟数据：文件夹‘scripts/a_generate_data’是模拟数据生成的代码
- 解析实际数据：文件夹‘scripts/b_parse_field_data’是解析实际数据的代码
- 预测其他数据：文件夹‘scripts/z_predict_new_data’是用于预测新实际数据代码，但在此之前，请解析数据
## 训练模型
在本研究中，我们直接以参数调参的方式，寻找最优训练超参

- 调参范式：文件夹‘scripts/ep_tuning_parameters’是调参的相关代码

## 模型测试
每次训练完成后是自动测试的，可查看‘models/cascade_method/train_run.py’文件，此代码是主程序代码。


## 已训练好的模型
训练好的模型文件保存在‘run_demo/data_model/model_checkpoint.pth’