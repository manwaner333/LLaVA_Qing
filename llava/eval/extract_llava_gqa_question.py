# 读取json 文件
import json
answers_file = 'playground/data/gqa/llava_gqa_testdev_balanced.json'
ans_file = open(answers_file, "w")
file_name = "data/gqa/questions1.2/testdev_balanced_questions.json"
with open(file_name, "r") as f:
    for line in f.readlines():
        dics = json.loads(line)
        for key, value in dics.items():
            question = value["question"]
            image = value["imageId"] + ".jpg"
            label = value["answer"]

            ans_file.write(json.dumps({"question_id": key,
                                       "text": question,
                                       "label": label,
                                       "image": image,
                                        }) + "\n")
            ans_file.flush()
ans_file.close()
