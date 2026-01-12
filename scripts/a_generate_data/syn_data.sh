# test
# local generate 
save_root=/home/htwang/Research/CascadeRMO/data/parsed

python models/generate_syn_data/syn_data_main.py --path_root $save_root/synthetic_data_ours_train --num_samples 1000 --num_curve 50 --start_id 1 
python models/generate_syn_data/syn_data_main.py --path_root $save_root/synthetic_data_ours_train --num_samples 1000 --num_curve 60 --start_id 1001 
python models/generate_syn_data/syn_data_main.py --path_root $save_root/synthetic_data_ours_train --num_samples 1000 --num_curve 80 --start_id 2001 
python models/generate_syn_data/syn_data_main.py --path_root $save_root/synthetic_data_ours_train --num_samples 1000 --num_curve 100 --start_id 3001 



python models/generate_syn_data/syn_data_main.py --path_root $save_root/synthetic_data_ours_valid --num_samples 500 --num_curve 60 --start_id 1 
python models/generate_syn_data/syn_data_main.py --path_root $save_root/synthetic_data_ours_valid --num_samples 500 --num_curve 80 --start_id 501 


python models/generate_syn_data/syn_data_main.py --path_root $save_root/synthetic_data_ours_test --num_samples 500 --num_curve 60 --start_id 1 
python models/generate_syn_data/syn_data_main.py --path_root $save_root/synthetic_data_ours_test --num_samples 500 --num_curve 80 --start_id 501 
# cd $save_root
# rm -r synthetic_data_ours_train synthetic_data_ours_valid synthetic_data_ours_test
# rm -r  synthetic_data_ours_test