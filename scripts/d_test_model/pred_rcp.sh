agc_list=31+51
dataset=syn_us_test
model_group=test_model_241231
for file in //home/htwang/result/cluster_seg_net/test_model_241231/*;
do
model_name=$(basename $file);
echo $model_name;
python models/cascade_method/test_run.py --model_group $model_group --model_name $model_name --test_set $dataset --save_num 5 --pp_min_len 5 --new_agc_list $agc_list --if_plot 1
done
