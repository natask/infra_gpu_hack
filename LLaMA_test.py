from transformers import AutoTokenizer, LlamaForCausalLM
from huggingface_hub import login
import time
import os
os.environ['HF_HOME'] = '/mount/model-cache'
os.environ['HF_HUB_CACHE'] = '/mount/model-cache'

model = LlamaForCausalLM.from_pretrained("casperhansen/Llama-3.3-70B-instruct-awq").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("casperhansen/Llama-3.3-70B-instruct-awq")

prompts = ["""say hi.""", """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order. Do this in python"""]
for prompt in prompts:
    prompt = f"""<|begin_of_text|>
<|system|>
You are a helpful assistant.
<|end_of_system|>

{prompt}

<|end_of_text|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(inputs.input_ids.numel())
    # Generate
    max_length = 2048
    num_beams = 4
    no_repeat_ngram_size = 3
    early_stopping = True
    for i in range(14):
        print(f"Max Length: {max_length}")
        start = time.time()
        generate_ids = model.generate(inputs.input_ids, max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping)
      
        print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        print(f"Time{time.time() - start}")
        print("*"*10)
