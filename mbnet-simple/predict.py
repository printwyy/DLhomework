import torch
from PIL import Image
from torchvision import transforms
from mobilenets import MobileNets

data_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
img = Image.open("./data/images/img-8-ship.png","r")
img = data_transform(img)
img = torch.unsqueeze(img,dim=0)
model = MobileNets(num_classes=10,large_img=False)
model.load_state_dict(torch.load("./saved_models/best_save_model.p"))
model.eval()

with torch.no_grad():
    output = model(img)
    pred = output.data.max(1, keepdim=True)[1]
    print(classification[pred[0][0]])


