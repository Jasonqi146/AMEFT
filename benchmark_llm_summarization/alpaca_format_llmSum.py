import json

file_path ="benchmark_llm_summarization/pairwise_evaluation_results.json"
# load processed marvel data set
with open(file_path, 'r') as file:
# Parse the JSON file and convert it into a Python dictionary
    data = json.load(file)
reslist = []
for item in data:
    context = item["article_text"]

    #shape new diction 
    current_dic = {}
    current_dic["article_id"] = item["article_id"] 
    current_dic["writer_id"] = item["writer_id"] 
    current_dic["evaluator_id"] = item["evaluator_id"] 
    current_dic["text-davinci-002_summary"] = item["text-davinci-002_summary"] 
    current_dic["overall_writer_better"] = item["overall_writer_better"] 
    current_dic["informative_writer_better"] = item["informative_writer_better"] 
    current_dic["instruction"]  = f"Summarize the following text: {context}"
    instruction = current_dic["instruction"]
    current_dic["input"] = ""
    current_dic["output"] = ""
    output = current_dic["output"]
    current_dic["expected_output"] = item["writer_summary"] 
    current_dic["text"] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {instruction}. ### Response: {output}"
    reslist.append(current_dic)

with open('benchmark_llm_summarization/alpaca_format.json', 'w') as f:
    json.dump(reslist, f, indent=4)

