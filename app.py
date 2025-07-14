from fastapi import FastAPI
from pydantic import BaseModel
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

app = FastAPI()

query_manager = ModelQueryManager("simple_fine_tuned_model")

class QueryRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

@app.get("/")
def read_root():
    return {"message": "Model API is running!"}

@app.post("/llm_query")
def query_model(request: QueryRequest):
    response = query_manager.query_basic(
        prompt=request.prompt,
        max_length=request.max_length,
        temperature=request.temperature
    )
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)