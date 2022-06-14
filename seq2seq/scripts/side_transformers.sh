# This scripts trains Adapters method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. For all datasets we tried
# with the adapter's bottleneck size of `task_reduction_factor`=[32, 16, 8], and report the 
# results on the test set for the model performing the best on the validation set.

folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

source scripts/env.sh

config_name=side_transformers
r=32

for g in 0
do
for lr in 3e-4
do
for seed in ${seeds[$2]}
do
    rm -r outputs/${config_name}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json seed int $seed
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json task_name str $2
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json eval_dataset_name str $2
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json test_dataset_name str $2
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json output_dir str outputs/${config_name}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json use_gate str "learnable"
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json task_reduction_factor int ${r}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json load_side_pretrained_weights str fisher-v2
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json learning_rate float ${lr}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json num_train_epochs int ${num_epochs[$2]}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json create_side_lm str2bool True
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json gate_alpha float ${g}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json add_residual_after str2bool True
    
    # python scripts/update_scripts_for_given_input.py configs/${config_name}.json encoder_side_layers eval "[0, 1, 2, 9, 10, 11]"
    # python scripts/update_scripts_for_given_input.py configs/${config_name}.json decoder_side_layers eval "[0, 1, 2, 9, 10, 11]"
    
    CUDA_VISIBLE_DEVICES=$1 TRANSFORMERS_OFFLINE=1 python run_seq2seq.py  configs/${config_name}.json

    cp outputs/${config_name}/all_results.json  all_output_logs/side_logit_transformers_after_r${r}_g${g}_fisher_v2_lr${lr}_$2@${seed}.json
done
done
done