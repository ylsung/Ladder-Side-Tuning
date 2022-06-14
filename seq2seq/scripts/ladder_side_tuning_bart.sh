
folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

source scripts/bart_env.sh

config_name=side_bart_encoder
r=8

lr=${learning_rates[$2]}

for seed in 0 1 2
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
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json num_train_epochs int 10
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json add_bias_sampling str2bool True
    
    CUDA_VISIBLE_DEVICES=$1 python run_classification.py  configs/${config_name}.json

    cp outputs/${config_name}/all_results.json  all_output_logs/side_bart_encoder_r${r}_g${g}_fisher_v2_lr${lr}_$2@${seed}.json
done