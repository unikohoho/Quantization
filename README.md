# Frequency-Aware PTQ for Medical Image Denoising

This repository contains the implementation associated with our MICCAI 2026 paper on post-training quantization for medical image denoising.

The project explores how low-bit post-training quantization can be applied to medical image restoration models while reducing quantization-induced degradation. The repository includes code organized for three representative restoration backbones:

* `NAFNet/`
* `SwinIR/`
* `Restormer/`

Each folder contains the corresponding model code, quantization modules, and scripts used for calibration and inference experiments.

## Repository Status

This repository is currently provided as an initial code release.
Additional documentation, cleaned configuration files, pretrained checkpoints, and detailed reproduction instructions will be updated after the official publication process is completed.

## Structure

```text
.
├── NAFNet/
├── SwinIR/
└── Restormer/
```

## Notes

* The current release focuses on the implementation of the quantization framework.
* Dataset files and pretrained model checkpoints are not included.
* Paths in the example scripts should be replaced with the user's local dataset and checkpoint paths.
* The official paper link will be added once it becomes available through the publisher.

## TODO

- [ ] Add detailed environment setup instructions
- [ ] Add step-by-step calibration and inference instructions
- [ ] Provide pretrained checkpoints
- [ ] Add official paper link after publication
- [ ] Add citation information after publication
