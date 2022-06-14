# The name of this experiment.
name=$2
lora_dim=64
seed=9595

for seed in 9596 9597
do
    # Save logs and models under snap/vqa; make backup.
    output=snap/vqa/${name}_d${lora_dim}@${seed}
    mkdir -p $output/src
    cp -r src/* $output/src/
    cp $0 $output/run.bash

    # export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP


    # See Readme.md for option details.
    CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
        unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/vqa.py \
        --distributed \
        --train train,nominival --valid minival  \
        --tqdm --output $output \
        --input_raw_images \
        --use_clip \
        --numWorkers 10 \
        --batchSize 32 --optim bert --lr 5e-4 --epochs 5 \
        --llayers 12 --xlayers 0 --rlayers 0 \
        --visualbert_style \
        --vqa_style_transform \
        --clip_model_name RN50x4 \
        --add_zero_padding \
        --gradient_accumulation_steps 8 \
        --loss_scale 500 \
        --warmup_ratio 0.05 \
        --report_step 400 \
        --use_separate_optimizer_for_visual \
        --sgd_lr 0.0001 \
        --sgd_momentum 0.0 \
        --schedule 2 \
        --use_positional_embedding \
        --pos_num 25 \
        --fp16 \
        --use_lora \
        --lora_dim ${lora_dim} \
        --clip_model_name RN50x4 \
        --loadLXMERTQA snap/pretrained/CLIP_VL_RN50x4 \
        --compute_time \
        --compute_memory \
        --seed ${seed} \
        ${@:5}  | tee $output/log.log

    CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/vqa.py \
    --distributed \
    --train train,nominival --valid minival  \
    --test test \
    --tqdm --output $output \
    --input_raw_images \
    --use_clip \
    --numWorkers 10 \
    --batchSize 32 --optim bert --lr 4e-5 --epochs 5 \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --visualbert_style \
    --vqa_style_transform \
    --clip_model_name RN50x4 \
    --add_zero_padding \
    --gradient_accumulation_steps 8 \
    --loss_scale 500 \
    --warmup_ratio 0.05 \
    --report_step 400 \
    --use_separate_optimizer_for_visual \
    --sgd_lr 0.001 \
    --sgd_momentum 0.0 \
    --schedule 2 \
    --use_positional_embedding \
    --pos_num 25 \
    --fp16 \
    --use_lora \
    --lora_dim ${lora_dim} \
    --clip_model_name RN50x4 \
    --load ${output}/BEST \
    ${@:5} 

    # The name of this experiment.
    name=$2

    # Save logs and models under snap/vqa; make backup.
    output=snap/gqa/${name}_d${lora_dim}@${seed}
    mkdir -p $output/src
    cp -r src/* $output/src/
    cp $0 $output/run.bash

    # export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP

    # See Readme.md for option details.
    CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
        unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/gqa.py \
        --distributed \
        --train train,valid --valid testdev \
        --tqdm --output $output \
        --input_raw_images \
        --use_clip \
        --numWorkers 10 \
        --batchSize 32 --optim bert --lr 5e-4 --epochs 5 \
        --llayers 12 --xlayers 0 --rlayers 0 \
        --visualbert_style \
        --vqa_style_transform \
        --loadLXMERTQA snap/pretrained/CLIP_VL_RN50x4 \
        --fp16 \
        --add_zero_padding \
        --gradient_accumulation_steps 8 \
        --warmup_ratio 0.05 \
        --report_step 400 \
        --use_separate_optimizer_for_visual \
        --sgd_lr 0.001 \
        --sgd_momentum 0.0 \
        --schedule 3 \
        --use_positional_embedding \
        --pos_num 25 \
        --clip_model_name RN50x4 \
        --loss_scale 500 \
        --use_lora \
        --lora_dim ${lora_dim} \
        --compute_time \
        --compute_memory \
        --seed ${seed} \
        ${@:5}  | tee $output/log.log


    # The name of this experiment.
    name=$2

    # Save logs and models under snap/vqa; make backup.
    output=snap/snli/${name}_d${lora_dim}@${seed}
    mkdir -p $output/src
    cp -r src/* $output/src/
    cp $0 $output/run.bash

    # export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP

    # See Readme.md for option details.
    CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
        unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/snli.py \
        --distributed \
        --train train --valid valid  \
        --tqdm --output $output \
        --input_raw_images \
        --use_clip \
        --numWorkers 10 \
        --batchSize 32 --optim bert --lr 5e-4 --epochs 2 \
        --llayers 12 --xlayers 0 --rlayers 0 \
        --visualbert_style \
        --vqa_style_transform \
        --clip_model_name RN50x4 \
        --loadLXMERT snap/pretrained/CLIP_VL_RN50x4 \
        --fp16 \
        --add_zero_padding \
        --gradient_accumulation_steps 8 \
        --report_step 400 \
        --warmup_ratio 0.05 \
        --use_separate_optimizer_for_visual \
        --sgd_lr 0.001 \
        --sgd_momentum 0.0 \
        --schedule 1 \
        --use_positional_embedding \
        --pos_num 25 \
        --clip_model_name RN50x4 \
        --use_lora \
        --lora_dim ${lora_dim} \
        --compute_time \
        --compute_memory \
        --seed ${seed} \
        ${@:5}  | tee $output/log.log

done
