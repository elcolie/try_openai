# python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path  /Users/sarit/Downloads/xxmix9realisticsdxl_testV20.safetensors --from_safetensors --dump_path ai_directory/xxmix9realisticsdxl_testV20
export MODEL_NAME="../ai_directory/xxmix9realisticsdxl_testV20"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="/Users/sarit/study/try_openai/try_fine_tune/folder/junk"

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 5 \
  --checkpointing_steps=5 \
  --output_dir="sdxl-sarit"
