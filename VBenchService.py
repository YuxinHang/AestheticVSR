"""
This is a util module to call VBench service to get aesthetic quality score.
The result is used in the new loss function of StableVSR.
Note: It takes ~14 seconds to finish the evaluation of rabbit video: /home/yuxin/video_evaluation/videos/a_rabbit_eating_flowers,_artstation_depth_generated.mp4
"""

import subprocess
import json

def getAestheticQualityScore(video_path, cuda_name, vbench_conda_env="vbench"):
    # inside: switch the conda env to vbench
    command = ["conda", "run", "-n", vbench_conda_env, "python", "/home/yuxin/VBench/getAestheticQuality.py", "--video_path", video_path, "--cuda_name", cuda_name]

    result = subprocess.run(command, capture_output=True, text=True, check=True)

    output = json.loads(result.stdout)

    # print("<=== output: ", output)
    # print(output.get("aestheticQualityScore"))

    return output.get("aestheticQualityScore")

# getAestheticQualityScore("/home/yuxin/video_evaluation/videos/a_rabbit_eating_flowers,_artstation_depth_generated.mp4")
