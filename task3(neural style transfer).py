import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# --------- CONFIG ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256  # Use smaller size on CPU

# --------- LOAD & TRANSFORM IMAGES ---------
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Replace with your image paths
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")

assert content_img.size() == style_img.size(), "Images must be the same size"

# --------- DISPLAY FUNCTION ---------
unloader = transforms.ToPILImage()

def show_image(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title: plt.title(title)
    plt.pause(0.001)

# --------- LOAD PRETRAINED VGG ---------
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# --------- CONTENT & STYLE LOSS CLASSES ---------
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    a, b, c, d = input.size()  # batch, channels, height, width
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# --------- BUILD MODEL ---------
cnn_layers = list(cnn.children())
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

content_losses = []
style_losses = []

model = nn.Sequential()
i = 0  # increment for conv layers

for layer in cnn_layers:
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f"conv_{i}"
    elif isinstance(layer, nn.ReLU):
        name = f"relu_{i}"
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = f"pool_{i}"
    elif isinstance(layer, nn.BatchNorm2d):
        name = f"bn_{i}"
    else:
        continue

    model.add_module(name, layer)

    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)

# Trim model after last content/style layer
for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
        break
model = model[:i + 1]

# --------- RUN STYLE TRANSFER ---------
input_img = content_img.clone()
optimizer = optim.LBFGS([input_img.requires_grad_()])

style_weight = 1e6
content_weight = 1

print("Optimizing...")
run = [0]
while run[0] <= 300:
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        loss = style_weight * style_score + content_weight * content_score
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Step {run[0]}:")
            print(f"Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")
        return loss

    optimizer.step(closure)

# --------- DISPLAY OUTPUT ---------
input_img.data.clamp_(0, 1)
show_image(input_img, title="Styled Image")
plt.show()
