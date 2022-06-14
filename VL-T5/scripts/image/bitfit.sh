task=multitask

# or bart
model="t5"

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
    batch_size=500
fi

echo $folder_prefix
echo $backbone

feature=RN101

for seed in 9595 9596 9597
do
lr=3e-3
name=LMbitfit+r8+ln_bs${batch_size}_image224_lr${lr}@${seed}
output=snap/${folder_prefix}_${task}/$name

CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$2 \
    --master_port=26747 \
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
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --use_bitfit \
    --use_tasks_prompts \
    --tasks "vqa,gqa,nlvr,caption" \
    --feature ${feature} --n_boxes 36 --downsample \
    --image_size "(224,224)" \
    --run_name $name \
    --seed ${seed}
done