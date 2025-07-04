## image classification using PyTorch

import torch
from torchvision import transforms
import requests
from PIL import Image
import gradio as gr

# using existing database but could use own later
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")

labels = response.text.split("\n")

def predict(inp):
    """ Predict what the image is.
    Original file gave really bad results, so looked to GPT for better
    parameters. Will see """

    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),                # Resize shortest side to 256
        transforms.CenterCrop(224),            # Crop the center 224x224
        transforms.ToTensor(),                 # Convert to tensor
        transforms.Normalize(                  # Normalize to ImageNet mean/std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Apply preprocessing for hopefully better results
    inp = preprocess(inp).unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        #process image to create prediction probabilities
        #softmax means proabilities of each class, allows confidence intervals per class
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        #store predictions in a dictionary.
        #keys are class labels, values are confidence probabilities
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}

    return confidences

""" Creating the Gradio Interface for predict function"""
gr.Interface(fn=predict, inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3)
       ).launch()
