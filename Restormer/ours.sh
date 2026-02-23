# 1. Quantization
python quantize_ours.py --bit 2 --gpu 2 \
    --model_path ./experiments/models/net_g_latest.pth \
    --train_lq /data1/uni/data/mayo2d/train/quarter_1mm \
    --save_dir ./experiments/test
 
# 2. Inference
python inference_ours.py --bit 2 --gpu 2 \
    --model_path ./experiments/test/Restormer_2bit_ours.pth \
    --output_dir /data1/uni/data/mayo2d/test/out/test




