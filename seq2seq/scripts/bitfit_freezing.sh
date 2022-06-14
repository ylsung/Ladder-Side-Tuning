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

file_name=bitfit

for encoder_list in '[6,7,8,9,10,11]' '[]'
do
for seed in 0 1 2
do
    rm -r outputs/${file_name}/
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json task_name str $2
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json eval_dataset_name str $2
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json test_dataset_name str $2

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json seed int $seed
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json num_train_epochs int ${num_epochs[$2]}
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json trainable_encoder_layers eval ${encoder_list}

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json trainable_decoder_layers eval None

    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  configs/${file_name}.json

    cp outputs/${file_name}/all_results.json  all_output_logs/${file_name}_e_${encoder_list}_$2@${seed}.json

done
done


for decoder_list in '[6,7,8,9,10,11]'
do
for seed in 0 1 2
do
    rm -r outputs/${file_name}/
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json task_name str $2
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json eval_dataset_name str $2
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json test_dataset_name str $2

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json seed int $seed
    python scripts/update_scripts_for_given_input.py configs/${file_name}.json num_train_epochs int ${num_epochs[$2]}

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json trainable_encoder_layers eval '[]'

    python scripts/update_scripts_for_given_input.py configs/${file_name}.json trainable_decoder_layers eval ${decoder_list}
    

    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  configs/${file_name}.json

    cp outputs/${file_name}/all_results.json  all_output_logs/${file_name}_e_d_${decoder_list}_$2@${seed}.json

done
done
