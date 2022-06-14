folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

source scripts/env.sh

config_name=network_pruning

r=8
lr=3e-3

for seed in 0 1 2
do
    rm -r outputs/${config_name}

    # python scripts/update_scripts_for_given_input.py configs/${config_name}.json model_name_or_path str t5-large
    # python scripts/update_scripts_for_given_input.py configs/${config_name}.json tokenizer_name str t5-large

    python scripts/update_scripts_for_given_input.py configs/${config_name}.json seed int $seed
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json task_name str $2
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json eval_dataset_name str $2
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json test_dataset_name str $2
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json output_dir str outputs/${config_name}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json use_gate str "one"
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json task_reduction_factor int ${r}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json load_side_pretrained_weights str fisher-v2
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json learning_rate float ${lr}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json num_train_epochs int ${num_epochs[$2]}
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json add_bias_sampling str2bool True
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json create_side_lm str2bool False
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json freeze_side_lm str2bool False
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json add_residual_after str2bool False
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json encoder_side_layers eval None
    python scripts/update_scripts_for_given_input.py configs/${config_name}.json decoder_side_layers eval None
    
    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  configs/${config_name}.json

    cp outputs/${config_name}/all_results.json  all_output_logs/network_pruning_lr${lr}_$2@${seed}.json
done