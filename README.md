# BLIP-2
BLIP-2 is an Vision Language model which consists of 3 models: a CLIP-like image encoder, a Querying Transformer (Q-Former) and a large language model. It excels in image-text interactions, offering capabilities like image captioning and visual question answering.

## Deploy BLIP-2 using Inferless:
- Deployment of BLIP-2 model using [transformers](https://github.com/huggingface/transformers).

---
## Prerequisites
- **Git**. You would need git installed on your system if you wish to customize the repo after forking.
- **Python>=3.8**. You would need Python to customize the code in the app.py according to your needs.
- **Curl**. You would need Curl if you want to make API calls from the terminal itself.

---
## Quick Start
Here is a quick start to help you get up and running with this template on Inferless.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **inferless-runtime-config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the `Add a custom model` button.

- Select `Github` as the method of upload from the Provider list and then select your Github Repository and the branch.
- Choose the type of machine, and specify the minimum and maximum number of replicas for deploying your model.
- Configure Custom Runtime ( If you have pip or apt packages), choose Volume, Secrets and set Environment variables like Inference Timeout / Container Concurrency / Scale Down Timeout
- Once you click “Continue,” click Deploy to start the model import process.
  
Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
          --header 'Content-Type: application/json' \
          --header 'Authorization: Bearer <your_api_key>' \
          --data '{
                "inputs": [
                    {
                    "name": "img_url",
                    "shape": [1],
                    "data": ["https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"],
                    "datatype": "BYTES"
                    },
                    {
                    "name": "prompt",
                    "optional": true,
                    "shape": [1],
                    "data": ["Explain the image."],
                    "datatype": "BYTES"
                    }
                ]
                }'
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](https://docs.inferless.com/model-import/input-output-schema) for more.

```python
def infer(self,inputs):
        img_url = inputs["img_url"]
        prompt = inputs.get("prompt")
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.
```python
def finalize(self):
    self.model = None
```

For more information refer to the [Inferless docs](https://docs.inferless.com/).
