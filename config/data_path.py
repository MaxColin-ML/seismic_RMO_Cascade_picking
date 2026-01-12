import os 

if 1:  # htwang pc path
    data_root='/home/htwang/Research/CascadeRMO/data/parsed'
    save_root='/home/htwang/Research/CascadeRMO/Cascade_RMO_results'
    
if 0:  # XJTU server path
    data_root='/data/wht/seismic/RCP'
    save_root='/home/htwang/result/Cascade_RMO_Results'

if 0:  # BGP server path
    data_root='/hwdata/data/liangshuaizhe_data/AI_Project_curvaturePicking/data/data_1st 2nd segy/parsed_datasets'
    save_root='/hwdata/data/liangshuaizhe_data/Al_Project_CurvaturePicking/data/data_1st_2nd_segy/results_folder'
    segy_root='/hwdata/data/liangshuaizhe_data/Al_Project_CurvaturePicking/data/data_1st_2nd_segy/source_segy_data_20241226'

 
data_dict = {
        'syn_us_train': os.path.join(data_root, 'synthetic_data_ours_train'),
        'syn_us_valid': os.path.join(data_root, 'synthetic_data_ours_valid'),
        'syn_us_test': os.path.join(data_root, 'synthetic_data_ours_test'),
        'bp': os.path.join(data_root, 'bp'),
        'sh': os.path.join(data_root, 'sh'),
        'jsh': os.path.join(data_root, 'jsh'),
        'yz': os.path.join(data_root, 'yz'),
        'es360': os.path.join(data_root, 'es360'),
        'wl1': os.path.join(data_root, 'wl1'),
        'wl2': os.path.join(data_root, 'wl2'),
        'wl3': os.path.join(data_root, 'wl3'),
        'wl4': os.path.join(data_root, 'wl4'),
        'wl5': os.path.join(data_root, 'wl5'),
        'other': os.path.join(data_root, 'other'),
        'test0801': os.path.join(data_root, 'test_0801'),
    }

