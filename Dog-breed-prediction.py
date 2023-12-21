import torch
import torchvision.transforms as transforms
import json
import gradio as gr
import torchvision.models as models
from torch import nn
from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
with open("model/id2vec.json", "r") as f:
    id2vec = json.load(f)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedClassifier, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(features, num_classes)

    def forward(self, input):
        return self.resnet(input)


model = DogBreedClassifier(120)
model = torch.load("model/dog-breed-identification.pth", map_location=device)


def predict(img):
    input = transform(img)
    with torch.no_grad():
        output = model(input.unsqueeze(0))
        predicted = torch.argmax(output, 1).item()
    return id2vec[str(predicted)]


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image of a dog"),
    outputs=gr.Text(label="Prediction"),
    title="Dog Breed Prediction App",
    allow_flagging='never',
    description="Upload an image to predict the dog breed.",
)

if __name__ == '__main__':
    app.launch()
