from pipeline.stablevsr_pipeline import StableVSRPipeline
from diffusers import DDPMScheduler, ControlNetModel
from accelerate.utils import set_seed
from PIL import Image
import os
import argparse
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pathlib import Path
import torch
import cv2  # Add this for video processing
import numpy as np
import sys

# Check if CUDA is available
# temporary: Choose a specific GPU (e.g., GPU 1) if available
# print(torch.cuda.device_count())
# device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
torch.cuda.empty_cache()
# device = torch.device("cuda:4")
# print(f"Using device: {torch.cuda.get_device_name(device)}")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# if torch.cuda.is_available():
#     current_device = torch.cuda.current_device()
#     print(f"Using GPU: {torch.cuda.get_device_name(current_device)} (ID: {current_device})")
# else:
#     print("CUDA is not available, using CPU.")

# torch.cuda.set_per_process_memory_fraction(0.9)  # Sets usage up to 90% of total GPU memory
# torch.backends.cuda.matmul.allow_tf32 = True  # Enables TensorFloat32 for performance
# torch.cuda.memory.set_per_process_memory_fraction(0.8)  # Adjust in MB

def center_crop(im, size=128):
    width, height = im.size   # Get dimensions
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2
    return im.crop((left, top, right, bottom))

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame)
        frames.append(pil_frame)
    cap.release()
    return frames

# get arguments
parser = argparse.ArgumentParser(description="Test code for StableVSR.")
parser.add_argument("--out_path", default='./StableVSR_results/', type=str, help="Path to output folder.")
parser.add_argument("--in_path", type=str, required=True, help="Path to input folder (containing sets of LR images).")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of sampling steps")
args = parser.parse_args()

print("Run with arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")

# set parameters
set_seed(42)
device = torch.device('cuda:7')

###======= pretrained model ======
model_id = 'claudiom4sir/StableVSR'
###======= Our trained model, change the path when necessary ======
controlnet_model = ControlNetModel.from_pretrained("/home/yuxin/StableVSR/experiments/model8_checkpoint-20000", subfolder='controlnet') # your own controlnet model
###======= TODO: Load the aesthetic model, change the path when necessary ======
aesthetic_model = torch.load("/home/yuxin/StableVSR/experiments/model8_checkpoint-20000/aesthetic_model.pt", map_location=device)
aesthetic_model = aesthetic_model.to(device)
aesthetic_model.eval()
print("<===aesthetic_model: ", aesthetic_model)

pipeline = StableVSRPipeline.from_pretrained(model_id, controlnet=controlnet_model)
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipeline.scheduler = scheduler
### Add the aesthetic_model in the pipeline
pipeline.aesthetic_model = aesthetic_model
pipeline = pipeline.to(device)
pipeline.enable_xformers_memory_efficient_attention()
of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
of_model.requires_grad_(False)
of_model = of_model.to(device)
# of_model = torch.nn.DataParallel(of_model) # Data parallel to make use of multiple GPUs

# Extract frames from video
frames = extract_frames_from_video(args.in_path)
print("Initial frame size: ", frames[0].width, frames[0].height)
# Upscale frames using StableVSR
frames = pipeline('', frames, num_inference_steps=args.num_inference_steps, guidance_scale=0, of_model=of_model).images
frames = [frame[0] for frame in frames]
print("Generated frame size: ", frames[0].width, frames[0].height)

# Save the generated frames as a video
output_video_path = args.out_path
frame_size = (frames[0].width, frames[0].height)
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, frame_size)
for frame in frames:
    frame_cv2 = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # Convert PIL to OpenCV format
    video_writer.write(frame_cv2)
video_writer.release()
print("MP4 video saved at:", output_video_path)


# iterate for every video sequence in the input folder
# seqs = sorted(os.listdir(args.in_path))
# for seq in seqs:
#     frame_names = sorted(os.listdir(os.path.join(args.in_path, seq)))
#     frames = []
#     for frame_name in frame_names:
#         frame = Path(os.path.join(args.in_path, seq, frame_name))
#         frame = Image.open(frame)
#         # frame = center_crop(frame)
#         frames.append(frame)

#     # upscale frames using StableVSR
#     frames = pipeline('', frames, num_inference_steps=args.num_inference_steps, guidance_scale=0, of_model=of_model).images
#     frames = [frame[0] for frame in frames]
    
#     # save upscaled sequences
#     seq = Path(seq)
#     target_path = os.path.join(args.out_path, seq.parent.name, seq.name)
#     os.makedirs(target_path, exist_ok=True)
#     for frame, name in zip(frames, frame_names):
#         frame.save(os.path.join(target_path, name))
