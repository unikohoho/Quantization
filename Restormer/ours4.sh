# 1. Quantization
python quantize_ours.py --bit 4 --gpu 3 \
    --model_path ./experiments/models/net_g_latest.pth \
    --train_lq <DATA_ROOT>/mayo2d/train/quarter_1mm \
    --save_dir ./experiments/quantized_models_new
 
# 2. Inference
python inference_ours.py --bit 4 --gpu 3 \
    --model_path ./experiments/quantized_models_new/Restormer_4bit_ours.pth \
    --output_dir <DATA_ROOT>/mayo2d/test/out/Restormer_4bit_ours_new




