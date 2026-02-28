# # 1. Quantization
# python quantize_ours.py --bit 8 --gpu 3 \
#     --model_path ./experiments/models/model_latest.pth \
#     --train_lq <DATA_ROOT>/mayo2d/train/quarter_1mm \
#     --save_dir ./experiments/quantized_models
 
# # 2. Inference
# python inference_ours.py --bit 8 --gpu 3 \
#     --model_path ./experiments/quantized_models/NAFNet_ours_8bit.pth \
#     --test_lq <DATA_ROOT>/mayo2d/test/quarter_1mm \
#     --output_dir <DATA_ROOT>/mayo2d/test/out/NAFNet_ours_8bit

# 1. Quantization
python quantize_ours.py --bit 2 --gpu 3 \
    --model_path ./experiments/models_moving/model_latest.pth \
    --train_lq <DATA_ROOT>/moving700/train/low \
    --save_dir ./experiments/quantized_models_moving
 
# # 2. Inference
python inference_ours.py --bit 2 --gpu 3 \
    --model_path ./experiments/quantized_models_moving/NAFNet_ours_2bit.pth \
    --test_lq <DATA_ROOT>/moving700/test/low \
    --output_dir <DATA_ROOT>/moving700/test/out/NAFNet_ours_2bit

