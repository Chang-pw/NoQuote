# A more comprehensive citation recommendation writing assistant design
# Using 8-GPU parallel processing and KV-cache–based acceleration for recommendations, enabling more efficient batch processing.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
  --input_path INPUT_PATH \
  --output_path OUTPUT_PATH \
  --world_size 8 --top_n 50 --threshold 0.7 \
  --n_alpha 0.8 \
  --p_alpha 0.1 \
  --r_alpha 0.1