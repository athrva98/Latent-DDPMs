import os
import random

import cv2 as cv
import numpy as np
import torch
import cv2
from models import SegNet
from collections import OrderedDict
from config import device, save_folder, imsize
from utils_ld import ensure_folder
from torchinfo import summary
from torch import nn
from torch.nn.functional import interpolate

def save_weights(model):
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if ".module" in k:
            name = k.replace(".module", "")  # remove module.
            new_state_dict[name] = v
        elif "module." in k:
            name = k.replace("module.", "")  # remove module.
            new_state_dict[name] = v
    torch.save(f='./latent_models/weights.pt',obj=new_state_dict)

def load_model(checkpoint_path='./latent_models/BEST_checkpoint.tar', print_summary=False):
    # checkpoint = '{}/BEST_checkpoint.tar'.format(save_folder)  # model checkpoint
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    save_weights(model)
    label_nbr = 3
    model = SegNet(label_nbr).to(device)
    state_dict = torch.load('./latent_models/weights.pt')
    model.load_state_dict(state_dict)
    model = model.to(device)
    if print_summary:
        summary(model, input_size=(1,3,256,256))
    return model

class LatentEncoderDecoder:
    def __init__(self, model, mode='eval', requires_grad=False):
        if mode=='eval':
            model.eval()
        if not requires_grad:
            for p in model.parameters():
                p.requires_grad = False
        self.model = model
        self.encoder_layers = list(model.children())[:-5]
        self.decoder_layers = list(model.children())[-5:]
        self.unpooled_shapes = [] # intermediate outputs
        self.indices = []
        
    def full_forward(self, inputs):
        with torch.no_grad():
            out = self.model.forward(inputs)
        return out
    
    def encoder_forward(self, inputs):
        batch_size = inputs.shape[0]
        with torch.no_grad():
            in_ = inputs
            for layer in self.encoder_layers:
                enc_out = layer(in_)
                if len(enc_out) == 3:
                    in_ = enc_out[0]
                    self.unpooled_shapes.append(enc_out[2])
                    self.indices.append(enc_out[1])
        in_ = torch.reshape(in_, (batch_size, 1, 128, 196))
        in_ = interpolate(in_, size=(32, 32), mode='bilinear').reshape(batch_size, 1, 32, 32)
        return in_ # latent embeddings from the encoder

    def decoder_forward(self, embeddings):
        # print(embeddings.shape)
        batch_size = embeddings.shape[0]
        embeddings = interpolate(embeddings, size=(128, 196), mode='bilinear').reshape(batch_size, 512, 7, 7)
        with torch.no_grad():
            self.unpooled_shapes.reverse()
            self.indices.reverse()
            out_ = embeddings
            for k, layer in enumerate(self.decoder_layers):
                out_ = layer(out_, self.indices[k], self.unpooled_shapes[k])
        return out_



def test():
    
    model = load_model() # loads a pre-trained autoencoder
    
    latent_model = LatentEncoderDecoder(model=model)

    test_path = '/home/athrva/Desktop/DiffusionModels/Autoencoder/data/train/'
    test_images = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.jpg')]

    num_test_samples = 10
    samples = random.sample(test_images, num_test_samples)

    imgs = torch.zeros([num_test_samples, 3, imsize, imsize], dtype=torch.float, device=device)

    ensure_folder('images')
    for i, path in enumerate(samples):
        # Read images
        img = cv2.imread(path)
        img = cv2.resize(img, (imsize, imsize))
        cv2.imwrite('images/{}_image.png'.format(i), img)

        img = img.transpose(2, 0, 1)
        assert img.shape == (3, imsize, imsize)
        assert np.max(img) <= 255
        img = torch.FloatTensor(img / 255.)
        imgs[i] = img

    imgs = torch.tensor(imgs)
    print(imgs.shape)
    with torch.no_grad():
        embeddings = latent_model.encoder_forward(imgs)
        preds = latent_model.decoder_forward(embeddings)
    
    for i in range(num_test_samples):
        out = preds[i]
        out = out.cpu().numpy()
        out = np.transpose(out, (1, 2, 0))
        out = out * 255.
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)
        # out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
        cv.imwrite('images/{}_out_new.png'.format(i), out)
       



if __name__ == '__main__':
    test()
