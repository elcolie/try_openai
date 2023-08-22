#export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="digiplay/majicMIX_realistic_v6"
export dataset_name="/Users/sarit/study/try_openai/try_fine_tune/folder/junk"

accelerate launch --mixed_precision=no  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=50 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sarit-model"
