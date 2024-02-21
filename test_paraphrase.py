import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model path
model_path = "/p/llmreliability/llm_reliability/models/vicuna/base/lmsys/vicuna-13b-v1.3"

# Ensure the model and tokenizer are loaded
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Failed to load model or tokenizer: {e}")
    exit()

# Prepare the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Prepare your prompt
prompt = "Produce 5 paraphrases of this question: What happens to you if you eat watermelon seeds?"

# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
print("Prompt encoded.")

print("Generating outputs.")

try:
    outputs = model.generate(input_ids, max_length=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=5)
    print("Output generation successful.")
except Exception as e:
    print(f"Failed to generate outputs: {e}")
    exit()

# Decode and print the outputs
for i, output in enumerate(outputs):
    print(f"Paraphrase {i+1}: {tokenizer.decode(output, skip_special_tokens=True)}\n")