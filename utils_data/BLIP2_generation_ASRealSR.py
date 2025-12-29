from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default='/media/ssd8T/cxn/omniscalesr/LSDIR_Train', help='the dataset you want to tag.') # 
parser.add_argument("--start_gpu", type=int, default=0, help='if you have 5 GPUs, you can set it to 0/1/2/3/4 when using different GPU for parallel processing. It will save your time.') 
parser.add_argument("--all_gpu", type=int, default=1, help='if you set --start_gpu max to 5, please set it to 5') 
args = parser.parse_args()

root_folder = args.root_path
print(root_folder)
input_folder = os.path.join(root_folder, 'gt')
blip_path = os.path.join(root_folder, 'blipv2_caption')
os.makedirs(blip_path, exist_ok=True)


processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)

name_list = os.listdir(input_folder)
name_list.sort()

start_num = args.start_gpu * len(name_list)//args.all_gpu
end_num = (args.start_gpu+1) * len(name_list)//args.all_gpu

print(f'===== process [{start_num}   {end_num}] =====')

for name in tqdm(name_list[start_num:end_num]):
    basename = name.split('.')[0]
    img_path = os.path.join(input_folder, name) 
    image =Image.open(img_path)

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    # print(generated_text)
    caption = f"{generated_text},"
    caption_save_path = blip_path + '/'+ f'{basename}.txt'
    f = open(f"{caption_save_path}", "w")
    f.write(caption)
    f.close()
