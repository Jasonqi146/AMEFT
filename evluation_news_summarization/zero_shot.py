from vllm import LLM, SamplingParams
import vllm
from vllm.lora.request import LoRARequest
import evaluate
import argparse
import json
from vllm.lora.request import LoRARequest
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='llama_7b', help='model name')
    parser.add_argument('--model_path', type=str, default='', help='using model path or name')
    parser.add_argument('--lora_path', type=str, default='',help='')
    parser.add_argument('--dataset_path', type=str, default='', help='dataset path')
    # parser.add_argument('--output_path', type=str, default='', help='output path')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p')
    parser.add_argument('--top_k', type=int, default=5, help='top_k') 
    parser.add_argument('--max_tokens', type=int, default=512, help='max_tokens')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='tensor parallel size')
    parser.add_argument('--use_lora', type=bool ,default=False,help='')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()

    model = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_lora=args.use_lora
    )

    with open('alpaca_news_summarization_test.json') as f:
        test_data = json.load(f)
    
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])

    # Write the model's answer and ground ture into file
    pairs_list = []
    # Test samples:
    # test_data = test_data[:2]
    # system_prompt = "In short paragraph, summarize the following text: "
    for item in test_data:
        current_pair = {}
        prompts = item['instruction'] + " " + item['input']
        conversations = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompts}],
            tokenize=False,
        )
        output = model.generate([conversations], sampling_params)
        # , lora_request=LoRARequest("sft_adapter", 1, args.lora_path))
        ground_true = item['output']
        current_pair['article'] = item['input']
        current_pair['model_output'] = output[0].outputs[0].text.strip().replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
        current_pair['ground_true'] = ground_true
        pairs_list.append(current_pair)

    
    with open('{}_news_summarization_results.json'.format(args.model_name), 'w') as f:
        json.dump(pairs_list, f, indent=4)


if __name__ == "__main__":
    main()