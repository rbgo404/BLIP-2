INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Explain the image."]
    },
    "img_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"]
    }
}