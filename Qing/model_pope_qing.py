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
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import random
import numpy as np
import pickle
from scipy.stats import entropy
import spacy

# np.random.seed(42)
# torch.manual_seed(42)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

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

    # idxs = ['COCO_val2014_000000310196', 'COCO_val2014_000000210789', 'COCO_val2014_000000429109', 'COCO_val2014_000000211674',
    #  'COCO_val2014_000000458338', 'COCO_val2014_000000283412', 'COCO_val2014_000000265719', 'COCO_val2014_000000461331',
    #  'COCO_val2014_000000544456', 'COCO_val2014_000000017708', 'COCO_val2014_000000574692', 'COCO_val2014_000000353180',
    #  'COCO_val2014_000000239444', 'COCO_val2014_000000569839', 'COCO_val2014_000000219622', 'COCO_val2014_000000300368',
    #  'COCO_val2014_000000482476', 'COCO_val2014_000000131115', 'COCO_val2014_000000157084', 'COCO_val2014_000000381895',
    #  'COCO_val2014_000000336872', 'COCO_val2014_000000075591', 'COCO_val2014_000000516916', 'COCO_val2014_000000542145',
    #  'COCO_val2014_000000218224', 'COCO_val2014_000000297078', 'COCO_val2014_000000033270', 'COCO_val2014_000000140583',
    #  'COCO_val2014_000000421455', 'COCO_val2014_000000288639', 'COCO_val2014_000000291936', 'COCO_val2014_000000063953',
    #  'COCO_val2014_000000526321', 'COCO_val2014_000000042190', 'COCO_val2014_000000553165', 'COCO_val2014_000000170517',
    #  'COCO_val2014_000000498759', 'COCO_val2014_000000360600', 'COCO_val2014_000000031773', 'COCO_val2014_000000500257',
    #  'COCO_val2014_000000574057', 'COCO_val2014_000000456178', 'COCO_val2014_000000565941', 'COCO_val2014_000000485564',
    #  'COCO_val2014_000000454642', 'COCO_val2014_000000205729', 'COCO_val2014_000000424792', 'COCO_val2014_000000329717',
    #  'COCO_val2014_000000012333', 'COCO_val2014_000000480122', 'COCO_val2014_000000515904', 'COCO_val2014_000000437347',
    #  'COCO_val2014_000000354229', 'COCO_val2014_000000538236', 'COCO_val2014_000000236865', 'COCO_val2014_000000217397',
    #  'COCO_val2014_000000060213', 'COCO_val2014_000000054025', 'COCO_val2014_000000084447', 'COCO_val2014_000000192660',
    #  'COCO_val2014_000000575755', 'COCO_val2014_000000354088', 'COCO_val2014_000000311327', 'COCO_val2014_000000456552',
    #  'COCO_val2014_000000350898', 'COCO_val2014_000000170365', 'COCO_val2014_000000021645', 'COCO_val2014_000000528905',
    #  'COCO_val2014_000000239347', 'COCO_val2014_000000007320', 'COCO_val2014_000000249715', 'COCO_val2014_000000080022',
    #  'COCO_val2014_000000564336', 'COCO_val2014_000000231589', 'COCO_val2014_000000465275', 'COCO_val2014_000000083275',
    #  'COCO_val2014_000000406403', 'COCO_val2014_000000131018', 'COCO_val2014_000000332625', 'COCO_val2014_000000332025',
    #  'COCO_val2014_000000318204', 'COCO_val2014_000000455157', 'COCO_val2014_000000069189', 'COCO_val2014_000000199764',
    #  'COCO_val2014_000000148766', 'COCO_val2014_000000175506', 'COCO_val2014_000000353096', 'COCO_val2014_000000207205']

    idxs = ['COCO_val2014_000000310196', 'COCO_val2014_000000210789', 'COCO_val2014_000000429109', 'COCO_val2014_000000211674']

    for line in tqdm(questions):
        idx = line["question_id"]
        if idx not in idxs:
            break
        image_file = line["image"]
        label = line["label"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
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
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        nlp = spacy.load("en_core_web_sm")
        sentences = []
        for item in nlp(outputs).sents:
            ele = item.text.strip()
            sentences.append(ele)

        # 添加相应的输出
        combined_hidden_states = None
        hidden_states = model_outputs['hidden_states']
        for ele in hidden_states:

            # if ele[-1].shape[1] != 1:
            #     hidden_state = ele[-1][:, -1:, :].detach().cpu().numpy().tolist()
            # else:
            #     hidden_state = ele[-1].detach().cpu().numpy().tolist()

            hidden_state = ele[-1][0].detach().cpu().numpy().tolist()

            if combined_hidden_states is None:
                combined_hidden_states = hidden_state
            else:
                combined_hidden_states = np.concatenate((combined_hidden_states, hidden_state), axis=0)

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
        token_entros = []
        tokens_idx = []
        token_and_logprobs = []
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
            token_and_logprobs.append([gen_tok, lp, entro])
            tokens_idx.append(gen_tok_id.detach().cpu().numpy().tolist())

        logprobs = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "tokens_idx": tokens_idx,
            "hidden_states": combined_hidden_states,
            "token_and_logprobs": token_and_logprobs
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
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.bin")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.5)  # 0.2
    parser.add_argument("--top_p", type=float, default=None)  # 0.99
    parser.add_argument("--top_k", type=int, default=None)  # 5 # there is no top-k before
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
