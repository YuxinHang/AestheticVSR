import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip
from PIL import Image
import os


# def generateMp4VideoFromFrames(frames, video_path, fps=10):
#     """
#     frames shape: a list of frames, where each frame has the shape [height, width, channel].
#     """
#     height, width = frames[0].shape[:2]
#     video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     for frame in frames:
#         video_writer.write(frame)
#     video_writer.release()

# def generateMp4FromBatchFrames(frames, video_path):
#     """
#     input: frames with the size [batch_size, temporal_frames, channels, height, width]
#     """
#     frames_list = []
#     for i in range(frames.shape[0]):
#         # Use only the central frame of the 3-frame sequence for each entry in the batch
#         frame_tensor = frames[i, :, 1, :, :]  # Select the middle frame (temporal dimension index 1)
#         frame_np = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Convert to numpy of shape [height, width, channels], and scale to 0-255
        
#         # if frames.shape[-1] != 256:  # Optionally resize if using low-res
#         #     frame_np = cv2.resize(frame_np, (256, 256))
        
#         frames_list.append(frame_np)
    
#     generateMp4VideoFromFrames(frames_list, video_path)
#     print(f"Saved video: {video_path}")

def generateMp4FromImages(images, video_path, fps=10):
    """
    input: images is a list of PIL.Image.Image file. 
    Example: [<PIL.Image.Image image mode=RGB size=512x512 at 0x7FF7DE555160>, <PIL.Image.Image image mode=RGB size=512x512 at 0x7FF7DE555190>, ...]
    """
    size = images[0].size
    
    # Initialize video writer with output path and settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, size)
    for img in images:
        # Convert the PIL image to a format compatible with OpenCV (numpy array)
        frame_np = np.array(img)
        # Convert RGB to BGR (OpenCV uses BGR format)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        # Write frame to the video
        video_writer.write(frame_bgr)

    # Release the video writer
    video_writer.release()

def generate_gif_from_mp4(mp4_path, gif_path, start_time=None, end_time=None, fps=10):
    """
    Generate a GIF from an MP4 file.

    Args:
        mp4_path (str): Path to the MP4 file.
        gif_path (str): Path to save the resulting GIF.
        start_time (float, optional): Start time in seconds. Defaults to the beginning of the video.
        end_time (float, optional): End time in seconds. Defaults to the end of the video.
        fps (int, optional): Frames per second for the GIF. Defaults to 10.
    """
    try:
        # Load the video file
        clip = VideoFileClip(mp4_path)
        
        # Trim the video if start_time and end_time are provided
        if start_time is not None or end_time is not None:
            clip = clip.subclip(start_time, end_time)
        
        # Write the GIF to the specified path
        clip.write_gif(gif_path, fps=fps)
        print(f"GIF saved successfully at {gif_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_frames_from_gif(gif_path, output_folder):
    """
    Extracts frames from a GIF and saves them as individual image files.

    Args:
        gif_path (str): Path to the input GIF file.
        output_folder (str): Path to the folder where frames will be saved.
    """
    try:
        # Open the GIF file
        gif = Image.open(gif_path)

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Iterate through the frames
        frame_number = 0
        while True:
            # Save the current frame as an image file
            frame_path = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
            gif.save(frame_path, format="PNG")
            print(f"Saved frame {frame_number} to {frame_path}")
            
            frame_number += 1
            
            # Move to the next frame
            try:
                gif.seek(gif.tell() + 1)
            except EOFError:
                # No more frames in the GIF
                break

    except Exception as e:
        print(f"An error occurred: {e}")

def extract_gif_and_frames_from_mp4(mp4_path, output_folder):
    gif_path = mp4_path.split(".")[0] + ".gif"
    generate_gif_from_mp4(mp4_path, gif_path)
    extract_frames_from_gif(gif_path, output_folder)


def batchConvertGifsToFrames(input_dir, output_base_dir):
    ### batch convert all gifs (e.g., cat.gif) under input_dir, 
    ### and create folders (e.g., cat/) under output_base_dir for the extracted frames.
    # Ensure the output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for gif_file in os.listdir(input_dir):
        if gif_file.endswith(".gif"):
            gif_path = os.path.join(input_dir, gif_file)
            # Create a folder named after the GIF (without extension) in the output directory
            folder_name = os.path.splitext(gif_file)[0]
            output_folder = os.path.join(output_base_dir, folder_name)
            print(f"Processing {gif_file} into {output_folder}...")
            extract_frames_from_gif(gif_path, output_folder)


# generate_gif_from_mp4("/home/yuxin/StableVSR/output_videos/a_rabbit_eating_flowers,_artstation_depth_generated_model7.mp4", "/home/yuxin/StableVSR/output_videos/a_rabbit_eating_flowers,_artstation_depth_generated_model7.gif")
# extract_frames_from_gif("/home/yuxin/StableVSR/output_videos/a_rabbit_eating_flowers,_artstation_depth_generated_model7.gif", "/home/yuxin/StableVSR/output_videos/model7_gif")
# extract_gif_and_frames_from_mp4("/home/yuxin/StableVSR/output_videos/a_rabbit_eating_flowers,_artstation_depth_generated_fromPretrainedModel.mp4", "/home/yuxin/StableVSR/output_videos/pretrained_frames")
# extract_gif_and_frames_from_mp4("/home/yuxin/StableVSR/input_videos/a_rabbit_eating_flowers,_artstation_depth.mp4", "/home/yuxin/StableVSR/evaluation_frames/input_frames")
# extract_gif_and_frames_from_mp4("/home/yuxin/StableVSR/output_videos/a_rabbit_eating_flowers,_artstation_depth_generated_model7.mp4", "/home/yuxin/StableVSR/evaluation_frames/output_frames")
# batchConvertGifsToFrames("/home/yuxin/StableVSR/input_real_videos", "/home/yuxin/StableVSR/evaluation_frames/input_frames")
# extract_gif_and_frames_from_mp4("/home/yuxin/StableVSR/output_videos/cat.mp4", "/home/yuxin/StableVSR/evaluation_frames/output_frames/cat")
extract_gif_and_frames_from_mp4("/home/yuxin/StableVSR/output_videos/woman.mp4", "/home/yuxin/StableVSR/evaluation_frames/output_frames/woman")
extract_gif_and_frames_from_mp4("/home/yuxin/StableVSR/output_videos/noodle.mp4", "/home/yuxin/StableVSR/evaluation_frames/output_frames/noodle")