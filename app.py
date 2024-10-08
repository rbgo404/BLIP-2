import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class InferlessPythonModel:

    def initialize(self):
        model_id = "Salesforce/blip2-flan-t5-xl"
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="cuda")

    def infer(self, inputs):
        img_url = inputs["img_url"]
        prompt = inputs.get("prompt")

        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = self.processor(raw_image, prompt,return_tensors="pt").to("cuda")

        output = self.model.generate(**inputs)
        output_text = self.processor.decode(output[0], skip_special_tokens=True).strip()
        
        return {"generated_output": output_text}

    def finalize(self):
        self.model = None
