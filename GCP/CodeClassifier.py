from gradio_client import Client

# Check HuggingFace Spces link for the model: https://huggingface.co/spaces/M-Abdelmegeed/Code-Classifier
def classify_code(input):
    client = Client("M-Abdelmegeed/Code-Classifier")
    result = client.predict(input)
    return result['label']