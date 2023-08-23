#https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README_sdxl.md
export MODEL_NAME="../ai_directory/xxmix9realisticsdxl_testV20"
export OUTPUT_DIR="sarit_sdxl_lora"
export DATASET_NAME="/Users/sarit/study/try_openai/try_fine_tune/folder/train"

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --gradient_checkpointing \
  --num_train_epochs=1000
