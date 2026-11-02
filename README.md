# MADF-Net_Review

Research code for multi-phase contrast-enhanced CT segmentation of the liver and liver tumors. This repository includes the MADF-Net implementation, data preprocessing scripts, slice-based training, volume-based evaluation, and comparison models.

The network takes arterial (ART/A), portal venous (PV/V), and delayed (DL/D) phase images as inputs and produces segmentation predictions for three classes: background, liver, and tumor.

## Repository Structure

```text
MADF-Net_Review/
├── bash/train_MADF-Net_MPLL.sh       # Original training launcher
├── config/                         # Configuration files for selected models
├── data_check/                     # HDF5 inspection utilities
├── dataset_prepare/                # Data conversion and sample-list generation
├── datasets/dataset_multiphase.py   # Multi-phase datasets and augmentation
├── images/proposed_model.png       # Architecture illustration
├── lists/lists_liver/               # Training and test sample lists
├── networks/MADF-Net.py             # Main model implementation: LiNet
├── networks/                       # Network components and comparison models
├── optimizers/                     # Learning-rate scheduling utilities
├── trainer/                        # Training routines
├── utils/                          # Losses, inference, and evaluation utilities
├── train_7metrics_MPLL_GY.py         # Training entry point
├── test_multiphase_everyepoch_7metrics_MPLL_JSC.py
│                                   # Checkpoint evaluation entry point
├── requirements.txt
└── LICENSE
```

## Environment Setup

Linux, Conda, and an NVIDIA GPU with CUDA support are recommended. The current training and evaluation routines call `.cuda()` directly and require a GPU environment.

### Clone the Repository

```bash
git clone https://github.com/jylEcho/MADF-Net_Review.git
cd MADF-Net_Review
```

### Create a Conda Environment

```bash
conda create -n madfnet python=3.10 -y
conda activate madfnet
```

### Install Dependencies

Install compatible versions of PyTorch and torchvision for your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install matplotlib tqdm einops pyyaml yacs
```

`requirements.txt` does not include every imported dependency. The additional command covers common imports used by the main entry points. Comparison models may require additional packages or pretrained weights.

These setup instructions were prepared through code inspection; a complete training run has not been performed to validate the environment.

## Data Preparation

### Original Data

Prepare registered three-phase CT volumes and segmentation masks with matching spatial dimensions and orientations.

The preprocessing scripts expect the following naming convention:

```text
data_nii.gz/
├── lesion-001-A.nii.gz
├── lesion-001-V.nii.gz
└── lesion-001-D.nii.gz

mask_correction/
└── lesion-001-V.nii.gz
```

Phase identifiers:

| Identifier | CT Phase |
| --- | --- |
| A | Arterial |
| V | Portal venous |
| D | Delayed |

Segmentation labels:

| Value | Class |
| --- | --- |
| 0 | Background |
| 1 | Liver |
| 2 | Tumor |

The preprocessing scripts read the portal venous phase mask and use the corresponding spatial annotations for all three phases.

Split the dataset at the patient level to prevent slices from the same patient appearing in both training and test sets.

### Preprocessing

Scripts in `dataset_prepare/` contain hard-coded absolute paths. Update their input and output settings before running them.

1. Prepare patient-level split lists; see `generate_patients_txt.py` for reference.
2. Configure `generate_liverbox.py` to generate the case-level liver bounding boxes required for training.
3. Update the image, mask, patient-list, bounding-box, and output paths in `rawdata_2D_train.py`.
4. Update the test patient list and input/output paths in `rawdata_2D_test.py`.
5. Update the data directories and list output directory in `get_txt.py`.

> **Warning:** The conversion scripts include logic that clears their output directories. Check all paths carefully and use dedicated preprocessing output folders.

After configuring the scripts, run:

```bash
python dataset_prepare/rawdata_2D_train.py
python dataset_prepare/rawdata_2D_test.py
python dataset_prepare/get_txt.py
```

By default, preprocessing clips CT intensities to `[-70, 180]` and normalizes them to `[0, 1]`. The training conversion script also retains liver-only slices with a configurable probability.

### Processed Data Structure

```text
processed/
├── train/
│   └── <slice_name>.npz
└── test/
    └── <case_name>.h5

lists/lists_liver/
├── train.txt
└── test_vol.txt
```

- Each training `.npz` file corresponds to one 2D slice.
- Each test `.h5` file corresponds to one case volume.
- `train.txt` contains NPZ filenames without extensions.
- `test_vol.txt` contains H5 filenames without extensions.

Required fields:

| File Type | Fields |
| --- | --- |
| Training `.npz` | `image_A`, `image_V`, `image_D`, `label`, `box`, `case_box` |
| Test `.h5` | `image_A`, `image_V`, `image_D`, `label`, `case_box`, `box_keys`, `box_values` |

The test preprocessing script uses the annotation directory to generate liver crop boxes by default. State the source of these boxes when reporting experiments. To evaluate a fully automatic pipeline, generate crop boxes from an independent coarse segmentation and report that setting.

## Training

Run the following command from the repository root, replacing the dataset paths:

```bash
CUDA_VISIBLE_DEVICES=0 python train_7metrics_MPLL_GY.py \
  --dataset Multiphase \
  --module networks.MADF-Net.LiNet \
  --model_name MADFNet \
  --root_path /path/to/processed/train \
  --test_path /path/to/processed/test \
  --list_dir ./lists/lists_liver \
  --output_dir ./model_out \
  --n_gpu 1 \
  --max_epochs 200 \
  --batch_size 6 \
  --img_size 224 \
  --base_lr 0.01 \
  --eval_interval 6 \
  --crop_train max_crop \
  --crop_inference standard_crop_inference \
  --is_savenii 0
```

Models are loaded dynamically from a string. The class defined in `networks/MADF-Net.py` is `LiNet`, so the corresponding model path is:

```text
networks.MADF-Net.LiNet
```

The original Bash launcher specifies `networks.MADF-Net.MADF-Net`, which does not match the class name. It also contains machine-specific absolute paths and a `cd ..` command. Update these before using that launcher.

### Main Arguments

| Argument | Description |
| --- | --- |
| `--root_path` | Directory containing training NPZ files |
| `--test_path` | Directory containing test H5 volumes |
| `--list_dir` | Directory containing sample lists |
| `--module` | Dynamically loaded model path |
| `--model_name` | Model identifier used by the training entry point |
| `--output_dir` | Output directory for results, logs, and weights |
| `--n_gpu` | Number of GPUs |
| `--max_epochs` | Number of training epochs |
| `--batch_size` | Batch size; check the trainer's scaling logic for multiple GPUs |
| `--img_size` | Input slice size |
| `--base_lr` | Initial learning rate |
| `--eval_interval` | Evaluation interval during training |
| `--crop_train` | Training crop strategy |
| `--crop_inference` | Inference crop strategy |
| `--is_savenii` | Whether to save NIfTI predictions during training: 0 or 1 |

### Optimization and Loss

The current multi-phase trainer uses SGD with:

- Momentum: `0.9`
- Weight decay: `0.0001`

The training loss combines weighted cross-entropy and Dice loss:

```text
Loss = 0.7 × WeightedCrossEntropy + 0.3 × DiceLoss
```

Cross-entropy class weights are:

```text
[0.3, 0.8, 4.5]
```

These values reflect the current implementation. Experiment notes in the launcher may describe different settings.

### Training Outputs

Training produces:

- Model checkpoints
- Text logs
- TensorBoard events
- Result tables and plots

See the training entry point and trainer for the exact output layout and checkpoint-saving schedule.

## Evaluation

The following entry point evaluates the `.pth` checkpoints in a directory and collects their results:

```bash
python test_multiphase_everyepoch_7metrics_MPLL_JSC.py \
  --dataset Multiphase \
  --module networks.MADF-Net.LiNet \
  --model_name MADFNet \
  --test_path /path/to/processed/test \
  --list_dir ./lists/lists_liver \
  --checkpoint_dir /path/to/checkpoints \
  --result_txt ./evaluation/all_checkpoints_result.txt \
  --img_size 224 \
  --crop_inference standard_crop_inference \
  --gpu 0
```

Use the same model architecture and input settings as in training.

### Evaluation Metrics

The evaluation pipeline records:

| Metric | Description |
| --- | --- |
| DSC | Dice similarity coefficient |
| Jaccard | Jaccard index |
| HD95 | 95th-percentile Hausdorff distance |
| ASSD | Average symmetric surface distance |
| RMSD | Root mean square surface distance |
| VOE | Volumetric overlap error |
| RVD | Relative volume difference |

The units of distance metrics depend on the voxel spacing used by the evaluation functions. Check the implementation before interpreting or reporting these values.

The evaluator loads weights with `strict=False` and reports missing or unexpected keys. Inspect these messages to verify that the checkpoint matches the architecture.

The result log is opened in overwrite mode. Use a new filename to preserve previous results.

## Reproducibility Notes

- The repository contains several experimental variants and comparison models. This README focuses on the current multi-phase MADF-Net workflow.
- Replace example paths and regenerate sample lists for your own dataset.
- The main model currently has a fixed three-class output. Changing `--num_classes` alone does not change its output layer.
- Dataset access information, pretrained weights, the formal paper citation, and benchmark results should be supplied by the maintainers.
- Training evaluates on `test_path`. Clearly distinguish the validation set used for checkpoint selection from the held-out test set used for final reporting.

## License

This repository includes an [MIT License](LICENSE).

Preserve existing copyright and license notices when using or redistributing the code. Third-party code, datasets, and pretrained weights remain subject to their respective terms.
