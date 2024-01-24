import os
import subprocess

# # Setting the GPU list or using the default value '0'
# gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', '0')
# gpulist = gpu_list.split(',')
#
# chunks = len(gpulist)
#
# ckpt = "llava-v1.5-7b"
# split = "testdev_balanced_questions"
# gqadir = "./playground/data/eval/gqa/data"
#
# processes = []
#
# for idx, gpu in enumerate(gpulist):
#     # Setting the CUDA_VISIBLE_DEVICES environment variable for each subprocess
#     env = os.environ.copy()
#     env['CUDA_VISIBLE_DEVICES'] = gpu
#
#     # Command to run the Python script
#     command = [
#         "python", "-m", "llava.eval.model_vqa_loader",
#         "--model-path", "liuhaotian/llava-v1.5-7b",
#         "--question-file", f"./playground/data/eval/gqa/data/questions1.2/{split}.json",
#         "--image-folder", "./playground/data/eval/gqa/data/images",
#         "--answers-file", f"./playground/data/eval/gqa/answers/{split}/{ckpt}/{chunks}_{idx}.jsonl",
#         "--num-chunks", str(chunks),
#         "--chunk-idx", str(idx),
#         "--temperature", "0",
#         "--conv-mode", "vicuna_v1"
#     ]
#
#     # Starting the subprocess
#     process = subprocess.Popen(command, env=env)
#     processes.append(process)
#
# # Waiting for all subprocesses to finish
# for process in processes:
#     process.wait()









