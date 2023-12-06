import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

print("Model loading...")

llm_path = "/home/mediboina.v/Vikash/medicalBot/LLM_HF"
tokenizer = LlamaTokenizer.from_pretrained(llm_path)

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}")

model = LlamaForCausalLM.from_pretrained(llm_path)
print("Model loaded")

def lama_inference(inputText, previousInput):
    global model
    model=model.to(device)
    prompt_template = f''' {previousInput}[INST]{inputText}[/INST]'''
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=4096)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    model=model.to('cpu')
    return output


# Example usage
# print(doctorOutput("I have been having a headache for 2 days.", ""))
