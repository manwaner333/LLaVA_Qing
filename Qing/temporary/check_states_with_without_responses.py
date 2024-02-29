import pickle
from tqdm import tqdm
import os
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import torch
import numpy as np
def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

if __name__ == "__main__":
    # path_with_response = f"../result/coco2014_val/answer_coco_pope_adversarial_new_prompt_without_response_1.bin"
    # with open(path_with_response, "rb") as f_with_response:
    #     res_with_response = pickle.load(f_with_response)
    #
    # path_without_response = f"../result/coco2014_val/answer_coco_pope_adversarial_new_prompt_without_response.bin"
    # with open(path_without_response, "rb") as f_without_response:
    #     res_without_response = pickle.load(f_without_response)
    #
    # for idx1, line1 in tqdm(res_with_response.items()):
    #     states_with_response = line1['logprobs']['hidden_states']
    #
    #
    # for idx2, line2 in tqdm(res_without_response.items()):
    #     states_without_response = line2['logprobs']['hidden_states']
    #
    # qingli = 3

    # model_path = os.path.expanduser("liuhaotian/llava-v1.6-mistral-7b")
    # model_name = get_model_name_from_path(model_path)
    # model_base = None
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    #
    # image_file = "COCO_val2014_000000310196.jpg"
    # # qs = "Provide a concise description of the given image."
    # qs = "Give me a description."
    # # qs = "Try you est to give the description."
    # cur_prompt = qs
    #
    # qs = "<image>" + '\n' + qs
    #
    # conv = conv_templates["llava_v1"].copy()
    # conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    #
    # input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
    #
    # image = Image.open(os.path.join("data/val2014", image_file))
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    #
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    #
    # with torch.inference_mode():
    #     model_outputs = model.generate(
    #         input_ids,
    #         images=image_tensor.unsqueeze(0).half().cuda(),
    #         do_sample=False,
    #         temperature=0,
    #         top_p=None,
    #         top_k=None,
    #         num_beams=1,
    #         # no_repeat_ngram_size=3,
    #         max_new_tokens=20,
    #         output_hidden_states=True,
    #         return_dict_in_generate=True,
    #         use_cache=True,
    #         output_scores=True)
    #
    # np.save("Qing/prompt_2", model_outputs['hidden_states'][0][-1].detach().cpu().numpy().tolist())

    prompt_1 = np.load("Qing/prompt_1.npy", allow_pickle=True)
    prompt_2 = np.load("Qing/prompt_2.npy", allow_pickle=True)
    # prompt_3 = np.load("Qing/prompt_3.npy", allow_pickle=True)
    #
    qingli = 3




