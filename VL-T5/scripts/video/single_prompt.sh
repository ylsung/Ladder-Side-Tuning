task=multitask_video

# or bart
model="bart"

echo $model

if [ $model == "t5" ]
then
    folder_prefix="VLT5"
    backbone="t5-base"
    batch_size=300
elif [ $model == "bart" ]
then
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=50
fi

echo $folder_prefix
echo $backbone

feature=ViT

lr=3e-4

name=${feature}_LMprompt_bs${batch_size}_image224_lr${lr}_subs_epoch7
output=snap/${folder_prefix}_${task}/$name

TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=26798 \
    src/${task}.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr ${lr} \
    --epochs 7 \
    --num_workers 4 \
    --backbone ${backbone} \
    --output $output ${@:2} \
    --num_beams 5 \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --encoder_prompt_len 40 \
    --mid_dim 800 \
    --use_single_prompt \
    --use_tasks_prompts \
    --tasks "tvqa,how2qa,tvc,yc2c" \
    --feature ${feature} --n_boxes 64 --downsample \
    --image_size "(224,224)" \
    --run_name $name
