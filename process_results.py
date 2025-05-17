import re
import jsonlines

def extract_boxed_numbers(text):
    def is_integer(s):
        try:
            int(s)
            return True
        except Exception as e:
            return False
    # 匹配 \boxed{...} 中的内容
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    if len(matches) <= 0:
        return []
    result = []
    match = matches[-1]
    numbers = [int(num.strip()) for num in match.split(',') if is_integer(num.strip())]
    return numbers

acc = []
precission = []
recall = []
with jsonlines.open("/mnt/maclabcv2/rubickjiang/codes/vllm_inference/output_llava-13B.jsonl", "r") as reader:
    for obj in reader:
        grounds = obj["ground"]
        grounds = [int(d) for d in grounds]
        pred = extract_boxed_numbers(obj["pred"])
        if len(pred) <= 0:
            continue
        pre_up = 0
        for d in pred:
            if d in grounds:
                pre_up += 1
        precission.append(pre_up/len(pred))
        recall_up = 0
        for d in grounds:
            if d in pred:
                recall_up += 1
        recall.append(recall_up/len(grounds))
        if len(pred) > 0 and pre_up/len(pred) >= 1 and recall_up/len(grounds) >= 1:
            acc.append(1)
        else:
            acc.append(0)

print("acc is: ", sum(acc)/len(acc))
print("precission is: ", sum(precission)/len(precission))
print("recall is: ", sum(recall)/len(recall))
print("Total matched items is: ", len(acc))
                
        
