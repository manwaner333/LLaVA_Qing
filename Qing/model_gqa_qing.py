import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from scipy.stats import entropy

from PIL import Image
import math
import numpy as np
import pickle
import spacy


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv = conv_templates[args.conv_mode].copy()  # 此处做了修改
        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file).replace('\\', '/')).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    responses = {}

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
                output_scores=True)

        output_ids = model_outputs['sequences']
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        nlp = spacy.load("en_core_web_sm")
        sentences = []
        for item in nlp(outputs).sents:
            ele = item.text.strip()
            sentences.append(ele)

        # extract the probability
        logits_list = []
        for ele in model_outputs.scores:
            logits_list.append(ele)
        logits = torch.stack(logits_list, dim=1)
        prob = torch.softmax(logits, dim=-1)[0]

        logprob = torch.log(prob)
        logprob = logprob.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()
        ent2 = 2 ** (entropy(prob, base=2, axis=-1))

        tokens = []
        token_logprobs = []
        top_logprobs = []
        tokens_idx = []
        token_entros = []
        k = 5

        for t in range(prob.shape[0]):
            gen_tok_id = output_ids[:, input_token_len + t]
            gen_tok = tokenizer.decode(gen_tok_id)
            lp = logprob[t, gen_tok_id]
            entro = ent2[t]

            sorted_indices = np.argsort(logprob[t, :])
            top_5_indices = sorted_indices[-k:]
            top_5_values = logprob[t, top_5_indices]

            top_logprob = {}
            for i in range(k):
                key = tokenizer.decode(top_5_indices[i])
                value = top_5_values[i]
                top_logprob[key] = value

            tokens.append(gen_tok)
            token_logprobs.append(lp)
            top_logprobs.append(top_logprob)
            token_entros.append(entro)
            tokens_idx.append(gen_tok_id)

        logprobs = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            'token_entros': token_entros,
            "tokens_idx": tokens_idx
        }

        usage = {
            "prompt_tokens": input_token_len,
            "completion_tokens": prob.shape[0],
            "total_tokens": input_token_len + prob.shape[0]
        }

        response_value = {"question_id": idx,
                          "prompts": cur_prompt,
                          "text": outputs,
                          "sentences": sentences,
                          "logprobs": logprobs,
                          "usage": usage,
                          "model_id": model_name
                          }
        responses[idx] = response_value

    with open(answers_file, 'wb') as file:
        pickle.dump(responses, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="data/gqa/images/images")
    parser.add_argument("--question-file", type=str, default="playground/data/gqa/llava_gqa_testdev_balanced_filter.json")
    parser.add_argument("--answers-file", type=str, default="result/gqa/output")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)  # 0
    parser.add_argument("--top_p", type=float, default=None)  # None
    parser.add_argument("--top_k", type=int, default=None)  # None
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=20)  # 128
    args = parser.parse_args()

    eval_model(args)
