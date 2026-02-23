# # 1. Quantization
# python quantize_ours.py --bit 8 --gpu 3 \
#     --model_path ./experiments/models/model_latest.pth \
#     --train_lq /data1/uni/data/mayo2d/train/quarter_1mm \
#     --save_dir ./experiments/quantized_models
 
# # 2. Inference
# python inference_ours.py --bit 8 --gpu 3 \
#     --model_path ./experiments/quantized_models/NAFNet_ours_8bit.pth \
#     --test_lq /data1/uni/data/mayo2d/test/quarter_1mm \
#     --output_dir /data1/uni/data/mayo2d/test/out/NAFNet_ours_8bit

# 1. Quantization
python quantize_ours.py --bit 2 --gpu 3 \
    --model_path ./experiments/models_moving/model_latest.pth \
    --train_lq /data1/uni/data/moving700/train/low \
    --save_dir ./experiments/quantized_models_moving
 
# # 2. Inference
python inference_ours.py --bit 2 --gpu 3 \
    --model_path ./experiments/quantized_models_moving/NAFNet_ours_2bit.pth \
    --test_lq /data1/uni/data/moving700/test/low \
    --output_dir /data1/uni/data/moving700/test/out/NAFNet_ours_2bit

