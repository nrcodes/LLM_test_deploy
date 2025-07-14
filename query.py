import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelQueryManager:
    def __init__(self, model_path: str = "simple_fine_tuned_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned model"""
        print(f"Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        print("Model loaded successfully!")
        return self.model, self.tokenizer

    def query_basic(self, prompt: str, max_length: int = 200, temperature: float = 0.7):
        """Basic query method using generate()"""
        if self.model is None:
            self.load_model()
        
        formatted_prompt = f"Human: {prompt}\nAssistant:"
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
if __name__ == "__main__":
    query_manager = ModelQueryManager("simple_fine_tuned_model")
    response = query_manager.query_basic("Can you provide more information about the firewall action taken on March 1, 2025, at 02:17?")

    print('Response! - ',response)