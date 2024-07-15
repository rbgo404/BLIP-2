import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class InferlessPythonModel:

    def initialize(self):
        model_id = "Salesforce/blip2-flan-t5-xl"
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    def infer(self, inputs):
        prompt = inputs["prompt"]  # Extract the prompt from the input
        img_url = inputs["img_url"]

        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        inputs = self.processor(raw_image, prompt, return_tensors="pt").to("cuda")

        out = self.model.generate(**inputs)
        return {"generated_result": self.processor.decode(out[0], skip_special_tokens=True).strip()}

    def finalize(self):
        self.model = None