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

    count = 0
    for line in tqdm(questions):
        idx = line["question_id"]
        # if count == 100:
        #     break
        image_file = line["image"]
        labels = line["new_labels"]
        sentences = line['sentences']
        qs = line["question"]
        res = line["response"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs + " " + res

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
            model_outputs = model(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                output_hidden_states=True,
                output_attentions=True)

        # hidden states
        hidden_states = model_outputs['hidden_states'][32][0]

        # logit
        logits = model_outputs['logits']

        # attention
        attentions = model_outputs['attentions']

        # ques_start_idx  and res_end_idx 在此处是相对于 input_ids的
        images_idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1].detach().cpu().numpy().tolist()[0]
        # print(images_idx)
        ques_start_idx = images_idx + 2 + 1  # 其中2对应 """ and '\n'
        input_len = input_ids.shape[1]
        res_end_idx = input_len - 4 - 1   # 其中6对应： "ASSISTANT:"
        ques_res_len = res_end_idx - ques_start_idx + 1
        shifted_input_ids = input_ids[:, ques_start_idx + 1:res_end_idx + 1]

        # 下面是相对于整个hidden states 里面的元素的
        before_image_tokens_length = 34
        image_tokens_length = 576
        ques_start_for_total_idx = before_image_tokens_length + image_tokens_length + 2         # -6 - ques_res_len + 1
        res_end_for_total_idx = ques_start_for_total_idx + ques_res_len - 1     # -6
        shifted_logits = logits[:, ques_start_for_total_idx:res_end_for_total_idx, :]

        # Convert logits to probabilities
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, 2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
        gathered_log_probs = gathered_log_probs.detach().cpu().numpy()

        # Convert logits to entropies
        probs = torch.softmax(shifted_logits, dim=-1)[0]
        probs = probs.detach().cpu().numpy()
        entropies = 2 ** (entropy(probs, base=2, axis=-1))

        tokens = []
        token_logprobs = []
        token_entropies = []
        tokens_idx = []
        token_and_logprobs = []
        for t in range(shifted_input_ids.shape[1]):
            gen_tok_id = shifted_input_ids[:, t]
            gen_tok = tokenizer.decode(gen_tok_id)
            lp = gathered_log_probs[:, t][0]
            entro = entropies[t]

            tokens.append(gen_tok)
            token_logprobs.append(lp)
            token_entropies.append(entro)
            token_and_logprobs.append([gen_tok, lp, entro])
            tokens_idx.append(gen_tok_id.detach().cpu().numpy().tolist())

        combined_attentions = {}
        combined_hidden_states = {}
        combined_token_logprobs = {}
        combined_token_entropies = {}

        ques_tokens = tokenizer(line["question"]).input_ids
        # print(ques_tokens)
        ques_tokens_len = len(ques_tokens) - 1
        ques_end_for_total_idx = ques_start_for_total_idx + ques_tokens_len - 1
        ques_hidden_states = hidden_states[ques_end_for_total_idx:ques_end_for_total_idx + 1, :]
        combined_hidden_states["ques"] = ques_hidden_states.detach().cpu().numpy().tolist()
        combined_attentions['ques'] = attentions[31][0, :, ques_start_for_total_idx:ques_end_for_total_idx + 1,
                                      ques_start_for_total_idx:ques_end_for_total_idx + 1].detach().cpu().numpy().tolist()

        sentences_end = []
        sentences_end.append(ques_end_for_total_idx)
        record = []
        start_idx = ques_end_for_total_idx + 1
        record1 = []
        for sent_i, sentence in enumerate(sentences):
            # sentence exist in the passage, so we need to find where it is [i1, i2]
            sentence_tf = "".join(sentence.split(" "))
            xarr = [i for i in range(len(tokens))]
            for i1 in xarr:
                mystring = "".join(tokens[i1:])
                if sentence_tf not in mystring:
                    break
            i1 = i1 - 1
            for i2 in xarr[::-1]:
                mystring = "".join(tokens[i1:i2 + 1])
                if sentence_tf not in mystring:
                    break
            i2 = i2 + 1

            sentence_len = i2 - i1 + 1
            sentence_end = start_idx + sentence_len - 1
            hidden_state = hidden_states[sentence_end:sentence_end + 1, :]
            attention = attentions[31][0, :, start_idx:sentence_end + 1, start_idx:sentence_end + 1].detach().cpu().numpy().tolist()
            combined_hidden_states[sent_i] = hidden_state.detach().cpu().numpy().tolist()
            combined_token_logprobs[sent_i] = token_logprobs[i1:i2+1]
            combined_token_entropies[sent_i] = token_entropies[i1:i2+1]
            combined_attentions[sent_i] = attention
            sentences_end.append(sentence_end)
            record.append([start_idx, sentence_end])
            record1.append([i1, i2])
            start_idx = sentence_end + 1
        log_probs = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "token_entropies": token_entropies,
            "tokens_idx": tokens_idx,
            # "ques_hidden_states": ques_hidden_states.detach().cpu().numpy().tolist(),
            # "res_hidden_states": res_hidden_states.detach().cpu().numpy().tolist(),
            "combined_hidden_states": combined_hidden_states,
            "combined_token_logprobs": combined_token_logprobs,
            "combined_token_entropies": combined_token_entropies,
            "combined_attentions": combined_attentions,
            "token_and_logprobs": token_and_logprobs
        }

        output = {"question_id": idx,
                  "image_file": line["image"],
                  "prompts": line["question"],
                  "text": res,
                  "sentences": sentences,
                  "logprobs": log_probs,
                  "labels": labels
                  }
        responses[idx] = output
        count += 1

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
