import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(fp):
    size = 512 if torch.cuda.is_available() else 128
    img = cv2.imread(fp) / 255.0
    img = cv2.resize(img, (size, size))
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    return img.to(device, torch.float)


def save_img(tensor, fp):
    img = tensor.cpu().clone().squeeze(0).detach().numpy().transpose((1, 2, 0)) * 255
    cv2.imwrite(fp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class ContentScore(nn.Module):
    def __init__(self, content_tensor):
        super(ContentScore, self).__init__()
        self.content = content_tensor.detach()
        self.loss_fn = nn.MSELoss()

    def forward(self, input):
        self.score = self.loss_fn(input, self.content)
        return input


class StyleScore(nn.Module):
    def __init__(self, style_tensor):
        super(StyleScore, self).__init__()
        self.style = StyleScore.gram(style_tensor).detach()
        self.loss_fn = nn.MSELoss()

    def forward(self, input):
        g = StyleScore.gram(input)
        self.score = self.loss_fn(g, self.style)
        return input

    @staticmethod
    def gram(tensor):
        batch, maps, h, w = tensor.size()
        features = tensor.view(batch * maps, h * w)

        gram_mat = torch.mm(features, features.t())
        return gram_mat.div(batch * maps * h * w)


class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, tensor):
        return (tensor - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)


def model_scores_init(content_img, style_img, content_layers=('conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'), style_layers=('conv_4')):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    model = nn.Sequential(Norm())
    content_scores = []
    style_scores = []
    i = 0
    for layer in cnn.children():
        if type(layer) == nn.Conv2d:
            i += 1
            name = f'conv_{i}'
        elif type(layer) == nn.ReLU:
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif type(layer) == nn.MaxPool2d:
            name = f'pool_{i}'
        else:
            name = f'bn_{i}'

        model.add_module(name, layer)

        if name in content_layers:
            content_score = ContentScore(model(content_img).detach())
            model.add_module(f'content_score_{i}', content_score)
            content_scores.append(content_score)
        if name in style_layers:
            style_score = StyleScore(model(style_img).detach())
            model.add_module(f'style_score_{i}', style_score)
            style_scores.append(style_score)

    # pruning extra layers
    for i in range(len(model) - 1, -1, -1):
        l_type = type(model[i])
        if l_type == ContentScore or l_type == StyleScore:
            break
    model = model[:(i + 1)]

    return model, content_scores, style_scores


def style_transfer(content_img, style_img, input_img, epochs=300, content_weight=1, style_weight=1000000):
    model, c_scores, s_scores = model_scores_init(content_img, style_img)

    input_img = nn.Parameter(input_img.data)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    i = 0
    while i < epochs:
        def callback():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            ss = sum([s.score for s in s_scores]) * style_weight
            cs = sum([c.score for c in c_scores]) * content_weight

            loss = ss + cs
            loss.backward()

            nonlocal i
            i += 1
            if i % 50 == 0:
                print(
                    f'epoch {i}: style loss - {ss.item()}, content loss - {cs.item()}')

            return ss + cs
        optimizer.step(callback)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == '__main__':
    style_img = load_image('starry_night.jpeg')
    content_img = load_image('sather.jpeg')
    input_img = content_img.clone()

    output_tensor = style_transfer(content_img, style_img, input_img)
    save_img(output_tensor, 'starry_sather.jpeg')
