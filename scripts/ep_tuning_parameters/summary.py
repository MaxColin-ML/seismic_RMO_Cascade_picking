import os
import pandas as pd
import itertools


root='/home/htwang/result/Cascade_RMO_Results'
save_root = 'scripts/ep_tuning_parameters/results'
result_list = []
group_list = ['tuning_train_para_250105']
for group in group_list:
    root_group = os.path.join(root, group)
    files = os.listdir(root_group)
    for file in files:
        csv_path = os.path.join(root_group, file, 'test_summary.csv')
        if os.path.exists(csv_path):
            test_metrics = pd.read_csv(csv_path)
            values = [[file, group] + row for row in test_metrics.values.tolist()]
            columns = list(test_metrics.columns)
            columns_df = ['models', 'group'] + columns 
            result_list += values
        xlsx_path = os.path.join(root_group, file, 'test_summary.xlsx')
        if os.path.exists(xlsx_path):
            test_metrics = pd.read_excel(xlsx_path)
            values = [[file, group] + row for row in test_metrics.values.tolist()]
            columns = list(test_metrics.columns)
            columns_df = ['models', 'group'] + columns 
            result_list += values

result_df = pd.DataFrame(result_list, columns=columns_df)
os.makedirs(save_root, exist_ok=True)
result_df.to_excel(os.path.join(save_root, 'tuning_result_details.xlsx'), index=False)
result_df.to_csv(os.path.join(save_root, 'tuning_result_details.csv'), index=False)

# the best parameter for each test data under each metric
datasets = ['bp', 'yz', 'sh', 'syn_us_test']
metrics = ['ASE', 'semblance', 'TR', 'semblance_manual', 'TASE', 'MSE_field']		

summary_result = []
for data, metric_name in itertools.product(datasets, metrics):
    df_data_flt = result_df[result_df['name'] == data][['models']+metrics]
    if metric_name == 'MSE_field':
        df_data_flt = df_data_flt.sort_values(by=[metric_name], ascending=True)
    else:
        df_data_flt = df_data_flt.sort_values(by=[metric_name], ascending=False)
    first_row = df_data_flt.iloc[0][metrics].values
    model = df_data_flt.iloc[0]['models']
    summary_result.append([data, metric_name] + list(first_row) + [model])

summary_df = pd.DataFrame(summary_result, columns=['name', 'metric'] + metrics + ['model'])
summary_df.to_excel(os.path.join(save_root, 'tuning_result_summary.xlsx'), index=False)