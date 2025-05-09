from transformers import AutoTokenizer, AutoModelForCausalLM
from safe_llm_finetune.fine_tuning.base import ModelAdapter


class Gemma_3_1B_it(ModelAdapter):
    def __init__(self):
        super().__init__("google/gemma-3-1b-it")
    
            
            
    def load_model(self, model_id: str):
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto"
        )

    def load_tokenizer(self, model_id: str):
        return AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True
        )

    def save_model(self, model, path: str):
        model.save_pretrained(path)

    def generate(self, model, tokenizer, prompt: str, **kwargs) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, **kwargs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
