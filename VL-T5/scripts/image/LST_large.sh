task=multitask

# or bart
model="t5"

echo $model

if [ $model == "t5" ]
then
    folder_prefix="VLT5"
    backbone="t5-large"
    batch_size=300
elif [ $model == "bart" ]
then
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=500
fi

echo $folder_prefix
echo $backbone

feature=RN101

for seed in 9595 9596 9597
do
lr=3e-3
name=fixed_large_${feature}_LMside4_wd3e-2_bs${batch_size}_image224_lr${lr}@${seed}
output=snap/${folder_prefix}_${task}/$name

CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$2 \
    --master_port=26672 \
    src/${task}.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr ${lr} \
    --epochs 20 \
    --num_workers 4 \
    --backbone ${backbone} \
    --output $output ${@:3} \
    --num_beams 5 \
    --weight_decay 3e-2 \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --use_side_transformers \
    --side_reduction_factor 4 \
    --decoder_side_layers "[1,3,5,7,9,11,13,15,17,19,21,23]" \
    --encoder_side_layers "[1,3,5,7,9,11,13,15,17,19,21,23]" \
    --load_side_pretrained_weights "fisher-v2" \
    --use_tasks_prompts \
    --tasks "vqa,gqa,nlvr,caption" \
    --feature ${feature} --n_boxes 36 --downsample \
    --image_size "(224,224)" \
    --run_name $name \
    --seed ${seed}
done