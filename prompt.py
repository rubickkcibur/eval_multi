import re
def format_prompts(question, steps):
    step_pattern = r"^Step\s+\d+\.\s*"
    enclosed_steps = []
    for idx, txt in enumerate(steps):
        refined_txt = re.sub(step_pattern, "", txt)
        enclosed_steps.append("<step_{}>\n{}\n</step_{}>".format(idx+1, refined_txt, idx+1))

    ret_text = "The following is an image understanding problem and a solution (split into steps, enclosed with tags and indexed from 1):\n[Problem]\n<image>\n{}\n[Solution]\n{}\nYour task is to review and critique the solution step by step. Once you identify errors in any step, return the index (indices) of the step(s) where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes \"not found\").\nPlease put your final answer (i.e., the index) in \\boxed{{}}. If there are multiple indices, please separate them with commas, such as \\boxed{{1, 3}}".format(question, "\n".join(enclosed_steps))
    return ret_text