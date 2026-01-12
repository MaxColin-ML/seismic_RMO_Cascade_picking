# wl1 
python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set wl1 --valid_range 50 --new_agc_list 15+31 --win_k 5 --bw_data 5 --bw_para 50 --min_len 15 --clu_eps 8

# wl2
python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set wl2 --valid_range 50 --new_agc_list 15+31 --win_k 5 --bw_data 5 --bw_para 50 --min_len 15 --clu_eps 8


# wl3
python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set wl3 --valid_range 50 --new_agc_list 15+31 --win_k 5 --bw_data 5 --bw_para 50 --min_len 15 --clu_eps 8

# wl4 
python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set wl4 --valid_range 50 --new_agc_list 15+31 --win_k 5 --bw_data 5 --bw_para 50 --min_len 15 --clu_eps 8


# wl5
python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set wl5 --valid_range 50 --new_agc_list 15+31 --win_k 5 --bw_data 5 --bw_para 50 --min_len 15 --clu_eps 8


# yz 
python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set yz --valid_range 50  --new_agc_list 15+31 --win_k 5 --bw_data 5 --bw_para 50 --min_len 10 --clu_eps 4 --samp_list_npy /home/htwang/research/seismic/RMO_picking/Cascade_RMO_Code/scripts/ep_comp_w_mp/log/samp_list_yz.npy


# # sh 
# python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set sh --valid_range 50  --new_agc_list 15+31 --win_k 5 --bw_data 5 --bw_para 50 --min_len 20 --clu_eps 8 --samp_list_npy /home/htwang/research/seismic/RMO_picking/Cascade_RMO_Code/scripts/ep_comp_w_mp/log/samp_list_sh.npy

# # bp 
# python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set bp --valid_range 50  --new_agc_list 9+15 --win_k 5 --bw_data 5 --bw_para 50 --min_len 20 --clu_eps 8 --samp_list_npy /home/htwang/research/seismic/RMO_picking/Cascade_RMO_Code/scripts/ep_comp_w_mp/log/samp_list_bp.npy

# # es360 
# python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set es360 --valid_range 50  --new_agc_list 9+15 --win_k 5 --bw_data 5 --bw_para 50 --min_len 20 --clu_eps 8 


# other
CUDA_VISIBLE_DEVICES=1 python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set other --valid_range 50  --new_agc_list 51+101 --win_k 5 --bw_data 5 --bw_para 50 --min_len 20 --clu_eps 8  --pred_num 32



CUDA_VISIBLE_DEVICES=1 python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set other --valid_range 50  --new_agc_list 101+151 --win_k 5 --bw_data 5 --bw_para 50 --min_len 20 --clu_eps 8  --pred_num 32

CUDA_VISIBLE_DEVICES=1 python models/cascade_method/predict_run.py --model_group train_best_250514 --model_name MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1 --proc_num 4 --device 0 --mark_name pred=v2 --test_set other --valid_range 50  --new_agc_list 151+201 --win_k 5 --bw_data 5 --bw_para 50 --min_len 20 --clu_eps 8  --pred_num 32



CUDA_VISIBLE_DEVICES=0 python models/cascade_method/predict_run.py --model_group train_best_250709 --model_name MSFSegNet-31+51-32-tanh-01111-bs8-lr1.0e-04-BCE-SGD-S4 --proc_num 2 --device 0 --mark_name pred=other --test_set test0801 --valid_range 25  --new_agc_list 5+11 --win_k 5 --bw_data 5 --bw_para 50 --min_len 20 --clu_eps 8  --pred_num 6