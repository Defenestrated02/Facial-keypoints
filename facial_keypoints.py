import torch
import sys
import torchvision
from torch import nn
import torchvision.transforms.v2 as T
from PIL import Image, ImageDraw

def get_image_tensor(image):
        transfrom = T.Compose(
          [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize((100, 100)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ]
        )
        orig_w, orig_h = image.size
        image_tensor = transfrom(image)
        return image_tensor, orig_w, orig_h

class MyModel(nn.Module):
  def __init__(self):
      super(MyModel, self).__init__()
      self.f = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2),
      nn.ReLU(),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Flatten(),
      nn.Linear(50176, 64),
      nn.ReLU(),
      nn.Linear(64, 28),
      )

  def forward(self, x):
      return self.f(x)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = Image.open(sys.argv[1]).convert('RGB')
    image_tensor, orig_w, orig_h = get_image_tensor(image)
    image_tensor = image_tensor[None,:,:,:]
    model = MyModel()
    model.load_state_dict(torch.load('facial_keypoints_model.pth', map_location=torch.device(device)))
    point_scaled = model(image_tensor.to(device))
    print(point_scaled)
    points = []
    i=0
    while i<28:
        x = [int(point_scaled[0][i])/(100/orig_h),int(point_scaled[0][i+1])/(100/orig_w)]
        i = i+2
        points.append(x)
    draw = ImageDraw.Draw(image)
    for pair in points:
         draw.circle(xy = (pair[0], pair[1]), 
            fill = (235, 64, 52),
            radius = 5)
    image.show
    image = image.save("face_new.jpg")
    print(points)
main()