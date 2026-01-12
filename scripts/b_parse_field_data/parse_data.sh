root_folder=/home/htwang/Research/CascadeRMO/data/segy
parse_root=/home/htwang/Research/CascadeRMO/data/parsed
python models/segy2npy/process_segy.py --segy_path $root_folder/BP.sgy --save_root $parse_root/bp
python models/segy2npy/process_segy.py --segy_path $root_folder/JSH.segy --save_root $parse_root/jsh
python models/segy2npy/process_segy.py --segy_path $root_folder/SH.segy --save_root $parse_root/sh
python models/segy2npy/process_segy.py --segy_path $root_folder/YZ.sgy --save_root $parse_root/yz

python models/segy2npy/process_segy.py --segy_path $root_folder/WL/data01-psdm-gathers_jl3d.segy --save_root $parse_root/wl1
python models/segy2npy/process_segy.py --segy_path $root_folder/WL/data03a-pbm-gathers.segy --save_root $parse_root/wl2
python models/segy2npy/process_segy.py --segy_path $root_folder/WL/data04-psdm_gathers-sea3d.segy --save_root $parse_root/wl3
python models/segy2npy/process_segy.py --segy_path $root_folder/WL/data02-psdm-gathers-sea2d.segy --save_root $parse_root/wl4
python models/segy2npy/process_segy.py --segy_path $root_folder/WL/data03b-pbm-gathers.segy --save_root $parse_root/wl5

python models/segy2npy/process_segy.py --segy_path $root_folder/others/CRP_MIG_NEW.sgy --save_root $parse_root/other


python models/segy2npy/process_segy.py --segy_path /home/htwang/Research/CascadeRMO/data/other/250801_test_data --save_root $parse_root/test_0801 --file_type bin --samp_num 101 --samp_width 128

# 241道101采样 [101,241]
# 