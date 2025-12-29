'''
 * OmniScaleSR: Unleashing Scale-Controlled Diffusion Prior for Faithful and Realistic Arbitrary-Scale Image Super-Resolution
'''
import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
import sys
sys.path.append('/media/ssd8T/cxn/OmniScaleSR')

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_omniscalesr import StableDiffusionControlNetPipeline
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.seemoredetail_model import SeemoRe

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_omniscalesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.omniscalesr_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.omniscalesr_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    return model

def load_blip_model(device='cuda'):
    
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
    model.eval()
    model.to(device)
    
    return model, processor
    
def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def get_blip_prompt(image, model, processor, device='cuda'):
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    validation_prompt = f"{generated_text}"
    return validation_prompt

def main(args, enable_xformers_memory_efficient_attention=True,):
    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("omniscalesr")

    pipeline = load_omniscalesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)
    blip_model, blip_processor = load_blip_model(accelerator.device)

    # init SeemoRe model
    sr4x_model = SeemoRe(num_experts = 3, 
                        embedding_dim = 36, 
                        use_shuffle = True, 
                        lr_space = 'exp', 
                        topk = 1
                        ).to(accelerator.device)
    pretrained_path = 'preset/models/seemoredetail_4x.pth'
    sr4x_model.load_state_dict(torch.load(pretrained_path)['params'])
    sr4x_model.eval()
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        else:
            image_names = [args.image_path]

        for image_idx, image_path in enumerate(image_names[:]):
            base = os.path.basename(image_path)
            stem = os.path.splitext(base)[0]

            print(f'================== process {image_idx} imgs... ===================')
            print(f'output: {args.output_dir}/sample00/{base}')

            # -------------------------
            # 1) Load image
            # -------------------------
            img_in = Image.open(image_path).convert("RGB")
            W0, H0 = img_in.size
            
            # -------------------------
            # 2) Prompt / conditioning
            # -------------------------
            _, ram_encoder_hidden_states = get_validation_prompt(args, img_in, model)

            prompt = get_blip_prompt(img_in, blip_model, blip_processor) + args.added_prompt
            neg_prompt = args.negative_prompt

            if args.save_prompts:
                os.makedirs(txt_path, exist_ok=True)
                txt_save_path = os.path.join(txt_path, f"{stem}.txt")
                with open(txt_save_path, "w") as f:
                    f.write(prompt)

            print(prompt)

            # -------------------------
            # 3) Resize / SR4x / align
            # -------------------------                
            lr_tensor = transforms.ToTensor()(img_in).unsqueeze(0).to(accelerator.device)
            with torch.no_grad():
                sr4x_tensor = sr4x_model(lr_tensor).clamp(0, 1)[0].detach().cpu()
            img_sr4x = transforms.ToPILImage()(sr4x_tensor)
            
            # build conditioning image for diffusion
            rscale = float(args.upscale)
            tgt_w = round(W0 * rscale)
            tgt_h = round(H0 * rscale)

            img_cond = img_sr4x.resize((tgt_w, tgt_h))
            img_cond = img_cond.resize((img_cond.size[0] // 8 * 8, img_cond.size[1] // 8 * 8))

            # -------- padding to suppressing edge artifacts --------
            pad = getattr(args, "pad", 16) 
            if pad > 0:
                cond_t = transforms.ToTensor()(img_cond).unsqueeze(0).to(accelerator.device)  # [1,3,H,W]
                cond_t = F.pad(cond_t, (pad, pad, pad, pad), mode="reflect")
                img_cond = transforms.ToPILImage()(cond_t[0].detach().cpu())

            # -------- align to divisible by 8 --------
                Wp, Hp = img_cond.size
                img_cond = img_cond.resize((Wp // 8 * 8, Hp // 8 * 8))
            # -------------------------------------------------------

            W, H = img_cond.size
            print(f'input size: {H}x{W} (pad={pad}) | target: {tgt_h}x{tgt_w}')

            # -------------------------
            # 4) Prepare output dirs & scalars
            # -------------------------
            for sample_idx in range(args.sample_times):
                os.makedirs(os.path.join(args.output_dir, f"sample{sample_idx:02d}"), exist_ok=True)

            sr_scale = torch.tensor([rscale], device=accelerator.device, dtype=torch.float32)
            lq_size  = torch.tensor([H0],   device=accelerator.device, dtype=torch.float32)
            
            pipe_kwargs = dict(
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                sr_scale=sr_scale,
                lq_size=lq_size,
                height=H,
                width=W,
                guidance_scale=args.guidance_scale,
                negative_prompt=neg_prompt,
                conditioning_scale=args.conditioning_scale,
                start_point=args.start_point,
                ram_encoder_hidden_states=ram_encoder_hidden_states,
                latent_tiled_size=args.latent_tiled_size,
                latent_tiled_overlap=args.latent_tiled_overlap,
                args=args,
            )

            # -------------------------
            # 5) Sampling
            # -------------------------
            
            for sample_idx in range(args.sample_times):  
                with torch.autocast("cuda"):
                    out_img = pipeline(prompt, img_cond, **pipe_kwargs).images[0]
                
                # -------- crop back padding (diffusion-only) --------
                if pad > 0:
                    ow, oh = out_img.size
                    out_img = out_img.crop((pad, pad, ow - pad, oh - pad))
                # ---------------------------------------------------
                
                if args.align_method == 'nofix':
                    out_img = out_img
                else:
                    if args.align_method == 'wavelet':
                        out_img = wavelet_color_fix(out_img, img_in)
                    elif args.align_method == 'adain':
                        out_img = adain_color_fix(out_img, img_in)


                out_img = out_img.resize((round(W0 * rscale), round(H0 * rscale)))
                    
                save_path = os.path.join(args.output_dir, f"sample{sample_idx:02d}", base)
                out_img.save(save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--omniscalesr_model_path", type=str, default=None)
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default=", clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=float, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr') # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point. 
    parser.add_argument("--save_prompts", action='store_true')
    args = parser.parse_args()
    main(args)



