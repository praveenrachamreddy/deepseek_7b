import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DeepSeekInference:
    def __init__(self, model_path):
        # Detect available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map=self.device
        )
        
    def generate_response(self, prompt, max_length=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_path = '/app/deepseek-model'
    inference = DeepSeekInference(model_path)
    
    while True:
        prompt = input("Enter your prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        response = inference.generate_response(prompt)
        print("Response:", response)

if __name__ == "__main__":
    main()
