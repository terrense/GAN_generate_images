import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
from model import Generator, Discriminator
from utils import get_data_loader

# 超参数设置
image_size = 64
latent_size = 100
hidden_size = 256
batch_size = 64
num_epochs = 100
learning_rate = 0.0002
data_dir = './data/vehicles/'  # 这里需要有你的车辆图像数据集

# Prepare data
data_loader = get_data_loader(image_size, batch_size, data_dir)

# Initialization of the networks
D = Discriminator(image_size * image_size * 3, hidden_size, 1).cuda()
G = Generator(latent_size, hidden_size, image_size * image_size * 3).cuda()

# Loss function and optimizor
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# Create the file folder the save new images
if not os.path.exists('./data/fake_images/'):
    os.makedirs('./data/fake_images/')

# Train GAN
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Get real images and their labels
        real_images = images.view(batch_size, -1).cuda()
        real_labels = torch.ones(batch_size, 1).cuda()
        
        # 生成假的图像及其标签
        z = torch.randn(batch_size, latent_size).cuda()
        fake_images = G(z)
        fake_labels = torch.zeros(batch_size, 1).cuda()

        # train discriminator
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        z = torch.randn(batch_size, latent_size).cuda()
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.4f}, D(G(z)): {fake_score.mean().item():.4f}')

    # After every epoch generate some images and save them
    fake_images = fake_images.view(fake_images.size(0), 3, image_size, image_size)
    fake_images = fake_images.data.cpu()
    vutils.save_image(fake_images, f'./data/fake_images/fake_images_{epoch+1}.png')

print("Training is Done!")
