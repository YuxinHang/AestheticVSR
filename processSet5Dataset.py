# from datasets import load_dataset
# from super_image import EdsrModel
# from super_image.data import EvalDataset, EvalMetrics
# import os
# import shutil

# dataset = load_dataset('eugenesiow/Set5', 'bicubic_x2', split='validation')
# print(dataset.cache_files)
# print("dataset: ", dataset)

# for example in dataset:
#     print("HR Path:", example['hr'])
#     print("LR Path:", example['lr'])

# # print(dataset[0])

# # # Iterate through all examples
# # for example in dataset:
# #     print(example)


# ### Save the dataset in the following directory: /home/yuxin/StableVSR/Datasets/Set5
# output_base_dir = "/home/yuxin/StableVSR/Datasets/Set5"

# for example in dataset:
#     # Extract the relative paths of the images (from 'Set5_HR' and 'Set5_LR_x2' folders)
#     hr_path = example['hr']
#     lr_path = example['lr']
    
#     # Extract the subfolder hierarchy
#     hr_subfolder = os.path.basename(os.path.dirname(hr_path))
#     lr_subfolder = os.path.basename(os.path.dirname(lr_path))
    
#     # Create directories for HR and LR images
#     hr_output_dir = os.path.join(output_base_dir, hr_subfolder)
#     lr_output_dir = os.path.join(output_base_dir, lr_subfolder)
#     os.makedirs(hr_output_dir, exist_ok=True)
#     os.makedirs(lr_output_dir, exist_ok=True)
    
#     # Copy the images to their respective directories
#     shutil.copy(hr_path, os.path.join(hr_output_dir, os.path.basename(hr_path)))
#     shutil.copy(lr_path, os.path.join(lr_output_dir, os.path.basename(lr_path)))

# print(f"Images have been saved with the hierarchy under {output_base_dir}")
