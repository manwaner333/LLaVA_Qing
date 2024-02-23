#!/bin/bash#!/bin/bash

TYPEA="adversarial"
TYPEB="random"
TYPEC="popular"

nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_0_$TYPEA.json >> output/output0.log 2>&1 &

wait

nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_1_$TYPEA.json >> output/output1.log 2>&1 &

wait

nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_2_$TYPEA.json >> output/output2.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_3_$TYPEA.json >> output/output3.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_4_$TYPEA.json >> output/output4.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_5_$TYPEA.json >> output/output5.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_6_$TYPEA.json >> output/output6.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_7_$TYPEA.json >> output/output7.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_8_$TYPEA.json >> output/output8.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEA.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_9_$TYPEA.json >> output/output9.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_0_$TYPEB.json >> output/output10.log 2>&1 &

wait

nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_1_$TYPEB.json >> output/output11.log 2>&1 &

wait

nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_2_$TYPEB.json >> output/output12.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_3_$TYPEB.json >> output/output13.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_4_$TYPEB.json >> output/output14.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_5_$TYPEB.json >> output/output15.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_6_$TYPEB.json >> output/output16.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_7_$TYPEB.json >> output/output17.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_8_$TYPEB.json >> output/output18.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEB.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_9_$TYPEB.json >> output/output19.log 2>&1 &


wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_0_$TYPEC.json >> output/output20.log 2>&1 &

wait

nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_1_$TYPEC.json >> output/output21.log 2>&1 &

wait

nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_2_$TYPEC.json >> output/output22.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_3_$TYPEC.json >> output/output23.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_4_$TYPEC.json >> output/output24.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_5_$TYPEC.json >> output/output25.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_6_$TYPEC.json >> output/output26.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_7_$TYPEC.json >> output/output27.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_8_$TYPEC.json >> output/output28.log 2>&1 &

wait
nohup python -u llava/eval/model_vqa.py --model-path liuhaotian/llava-v1.5-13b --question-file playground/data/coco2014_val_qa_eval/coco_pope_$TYPEC.json --image-folder data/val2014 --answers-file result/coco2014_val/answer_coco_pope_13b_9_$TYPEC.json >> output/output29.log 2>&1 &
