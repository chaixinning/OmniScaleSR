## OmniScaleSR: Scale-Controlled Diffusion for Faithful and Realistic Arbitrary-Scale Image Super-Resolution

### 📌 TODO
- [x] ~~Repo release~~
- [x] ~~Test codes release~~
- [x] Update paper link
- [x] Pretrained models
- [x] Training Code release

## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/chaixinning/OmniScaleSR.git
cd OmniScaleSR

# create an environment with python >= 3.10
conda create -n omniscalesr python=3.10
conda activate omniscalesr
pip install -r requirements.txt
```

## ⚡ Quick Inference
#### Step 1: Download the pretrained models
- Download the pretrained SD-2-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).
- Download the OmniScaleSR models [TBC]

#### Step 2: Prepare testing data
You can put the testing images in the `preset/datasets/test_datasets`.

#### Step 3: Running testing command
```
python test_omniscalesr.py \
--scale 16
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt '' \
--test_omniscalesr_model_path preset/models/test_omniscalesr \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output \
--num_inference_steps 50 \
--guidance_scale 7.5 \
--process_size 512 
```
