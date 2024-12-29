import pandas as pd
import numpy as np
import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from monai.transforms import Compose, ScaleIntensity, Resize, EnsureType
import csv

##############################################################
# 데이터 준비
##############################################################
adni_merge_path = "/home/guest2/Demographic.csv"
adni_data = pd.read_csv(adni_merge_path)
adni_data["DX_bl"] = adni_data["DX_bl"].str.strip()

ADNI_DIR = "/research/public-dataset/02-ADNI/02-output/"
MICAPIPE_DIR = ADNI_DIR + "micapipe/"

matching_records = adni_data.reset_index(drop=True)
matching_records['Index'] = matching_records.index

# TODO: 아래 경로 구성 부분은 기본 제공. 필요시 수정 가능
matching_records['ADNI_MRI_PATH'] = MICAPIPE_DIR + 'sub-ADNI' + matching_records['PTID'] + '/' + \
                                     'ses-M00' + \
                                     '/xfm/sub-ADNI' + matching_records['PTID'] + '_' + \
                                     'ses-M00' + \
                                     '_from-nativepro_brain_to-MNI152_2mm_mode-image_desc-SyN_Warped.nii.gz'

matching_records['MRI_EXISTS'] = matching_records['ADNI_MRI_PATH'].apply(lambda x: os.path.exists(x))
print(matching_records['MRI_EXISTS'])


##############################################################
# 중앙 슬라이스 추출 함수
# nibabel로 NIfTI 로드 후 z축 중앙 슬라이스
##############################################################
def get_center_axial_slice(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    z_center = data.shape[2] // 2
    slice_img = data[:, :, z_center]
    # (H,W) -> (C,H,W), C=1
    slice_img = slice_img[np.newaxis, ...]
    return slice_img


##############################################################
# Monai transform을 이용해 이미지 (128x128) normalize
# MRIDataset 정의
#
# Hint:
#  - MRI_EXISTS==True인 행만 사용
#  - get_center_axial_slice() 이용
#  - monai_transforms (ScaleIntensity, Resize 등) 사용
##############################################################

# [TODO] Image 를 Monai 를 활용하여 모델에 넣기 좋게 transform 을 진행한다
image_size = 128
monai_transforms = Compose([
    ScaleIntensity(),
    Resize((image_size, image_size)),
    EnsureType()
                            ])

# [TODO] Class 를 활용한, MRI 데이터를 호출
# VAE 코드에서 MRI 데이터 호출 코드 참고
class MRIDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records[records['MRI_EXISTS'] == True]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records.iloc[idx]
        mri_path = record['ADNI_MRI_PATH']
        img = get_center_axial_slice(mri_path)
        if self.transform:
            img = self.transform(img)
        return img

# [TODO] Train/Val split
# 힌트: 0.8 비율로 train, 나머지 val


train_size = int(0.8 * len(matching_records))
val_size = len(matching_records) - train_size
train_dataset, val_dataset = random_split(
    MRIDataset(matching_records, monai_transforms), 
    [train_size, val_size]
    )


batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


##############################################################
# 간단한 DCGAN Generator / Discriminator 정의

# [Hint]: ConvTranspose2d 에서 출력 크기 Hout, Wout 은 다음과 같이 계산됨
'''
# Hout = (Hin - 1) x stride - 2 x padding + kernel_size + output_padding
# Wout = (Win - 1) x stride - 2 x padding + kernel_size + output_padding
'''
# [TODO] ngf, ndf, nz, epoch 등 파라미터 계산하여 실제 모델 파라미터 구현 및 변경해보기
##############################################################
nz = 100  # latent vector size
ngf = 64  # generator feature size
ndf = 64  # discriminator feature size
nc = 1    # grayscale channel

'''
 기본 구조는 다음과 같습니다.

     ngf 는 Generator 네트워크의 각 레이어에서 사용되는 채널 수(피처 맵 수)를 결정하기 위한 기준 파라미터 임
     ngf 자체는 특정 레이어의 기본 채널 수를 의미하고, ngf*8, ngf*4, ngf*2와 같이 곱하는 것은 레이어별로 채널 수를 단계적으로 늘리거나 줄여나가기 위한 방식'

     # Generate 네트워크 기본 1 Block 의 구조 예시
     nn.ConvTranspose2d(nz, ngf*8, Kernel_size, stride, padding, bias=False),
     nn.BatchNorm2d(ngf*8),
     nn.ReLU(True),
     nn.Sigmoid()

     # Discriminator 네트워크 기본 1 Block 의 구조 예시
     nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
     nn.BatchNorm2d(ndf*2),
     nn.LeakyReLU(0.2, inplace=True),

'''

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Kernel_size = 4, stride = 1, padding = 0 
            nn.ConvTranspose2d(nz, ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # Kernel_size = 4, stride = 2, padding = 1 
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # Kernel_size = 4, stride = 2, padding = 1 
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # Kernel_size = 4, stride = 2, padding = 1 
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Kernel_size = 4, stride = 4, padding = 3 
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=4, padding=3, bias=False),
            # Sigmoid()
            nn.Sigmoid()
            # 최종 출력 크기: (nc, 128, 128)

        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            # 이미지 입력 크기: (nc, 128, 128)
            # Kernel_size = 4, stride = 2, padding = 1
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # Kernel_size = 4, stride = 2, padding = 1
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # Kernel_size = 4, stride = 2, padding = 1
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
            # 최종 출력 크기: (ndf*2, 32, 32)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.final_conv = nn.Conv2d(ndf*4, 1, 1, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        x = self.avgpool(x)
        x = self.final_conv(x)
        return self.sigmoid(x).view(-1,1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(nc, ndf).to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

fixed_noise = torch.randn(16, nz, 1, 1, device=device)

##############################################################
# [TODO] Training Loop
#
# - real_label=1, fake_label=0
# - D: real_data와 fake_data 구분 학습
# - G: fake_data를 real처럼 만들기
##############################################################

#Find Optimal parameter
num_epochs = 100

train_D_losses = []
train_G_losses = []
val_D_losses = []
val_G_losses = []

for epoch in range(num_epochs):
    netG.train()
    netD.train()
    total_D_loss = 0
    total_G_loss = 0
    for i, real_data in enumerate(train_loader):
        # real_data: (B,1,128,128)
        real_data = real_data.to(device)
        b_size = real_data.size(0)
        
        # [설명] Label 준비
        # real_label=1, fake_label=0을 사용해 Discriminator가 Real과 Fake를 구분하도록 유도
        real_label = torch.ones(b_size, 1, device=device)
        fake_label = torch.zeros(b_size, 1, device=device)

        # [TODO] Update D (Discriminator)
        # Hint:
        # 1) real_data를 netD에 넣어 Real인지 판단 후 loss 계산
        #       - [Hint]: netD(real_data)
        # 2) noise로 fake_data 생성, netD에 넣어 Fake 판단 후 loss
        #       - [Hint]: Loss 계산 방법 criterion(output_real, real_label) nn.BCELoss()
        # 3) D_loss = D_real_loss + D_fake_loss
        
        optimizerD.zero_grad()
        # 1)
        output = netD(real_data)
        D_real_loss = criterion(output, real_label)
        # 2)
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_data = netG(noise)
        output = netD(fake_data.detach())
        D_fake_loss = criterion(output, fake_label)
        # 3)
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizerD.step()

        # [TODO] Update G (Generator)
        # Hint:
        # 1) G는 fake_data를 real처럼 보이게 만들고 싶음
        #       - [Hint] netD(fake_data) # fake_data로 D 예측
        # 2) fake_data를 netD에 넣어 real_label과 비교
        # 3) G_loss = criterion(D(fake_data), real_label)
        # 4) G_loss.backward() -> optimizerG.step()

        optimizerG.zero_grad()
        # 1) 
        output = netD(fake_data)
        # 2)
        G_loss = criterion(output, real_label)

        G_loss.backward()
        optimizerG.step()

        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()

    avg_D_loss = total_D_loss / len(train_loader)
    avg_G_loss = total_G_loss / len(train_loader)

    # Validation
    netG.eval()
    netD.eval()
    val_D_loss_total = 0
    val_G_loss_total = 0
    with torch.no_grad():
        for val_real_data in val_loader:
            val_real_data = val_real_data.to(device)
            vb_size = val_real_data.size(0)
            real_label = torch.ones(vb_size,1,device=device)
            fake_label = torch.zeros(vb_size,1,device=device)

            # D val loss
            output = netD(val_real_data)
            val_D_real_loss = criterion(output, real_label)
            noise = torch.randn(vb_size, nz, 1, 1, device=device)
            val_fake_data = netG(noise)
            output = netD(val_fake_data)
            val_D_fake_loss = criterion(output, fake_label)
            val_D_loss = val_D_real_loss + val_D_fake_loss
            val_D_loss_total += val_D_loss.item()

            # G val loss
            val_output = netD(val_fake_data)
            val_G_loss = criterion(val_output, real_label)
            val_G_loss_total += val_G_loss.item()

    val_avg_D_loss = val_D_loss_total / len(val_loader)
    val_avg_G_loss = val_G_loss_total / len(val_loader)

    train_D_losses.append(avg_D_loss)
    train_G_losses.append(avg_G_loss)
    val_D_losses.append(val_avg_D_loss)
    val_G_losses.append(val_avg_G_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train D: {avg_D_loss:.4f}, G: {avg_G_loss:.4f} | Val D: {val_avg_D_loss:.4f}, G: {val_avg_G_loss:.4f}")

    # [TODO] 주기적으로 fake 이미지 샘플 저장
    if (epoch+1) % 2 == 0:
        with torch.no_grad():
            sample_fake = netG(fixed_noise).cpu().numpy()
        plt.figure(figsize=(10,10))
        for idx in range(min(16, sample_fake.shape[0])):
            plt.subplot(4,4,idx+1)
            plt.imshow(sample_fake[idx,0], cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Generated Samples Epoch {epoch+1}')
        plt.tight_layout()
        plt.savefig(f'generated_epoch_{epoch+1}.png')
        plt.close()

##############################################################
# Loss 곡선 Plot
##############################################################
plt.figure(figsize=(10,5))
plt.plot(train_D_losses, label='Train D Loss')
plt.plot(val_D_losses, label='Val D Loss')
plt.plot(train_G_losses, label='Train G Loss')
plt.plot(val_G_losses, label='Val G Loss')
plt.title('GAN Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('gan_loss_curve.png')
plt.close()

##############################################################
# CSV로 저장
##############################################################
with open('gan_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_D_loss', 'train_G_loss', 'val_D_loss', 'val_G_loss'])
    for i in range(num_epochs):
        writer.writerow([i+1, train_D_losses[i], train_G_losses[i], val_D_losses[i], val_G_losses[i]])

##############################################################
# Inference 결과
##############################################################
with torch.no_grad():
    sample_fake = netG(fixed_noise).cpu().numpy()
plt.figure(figsize=(10,10))
for idx in range(min(16, sample_fake.shape[0])):
    plt.subplot(4,4,idx+1)
    plt.imshow(sample_fake[idx,0], cmap='gray')
    plt.axis('off')
plt.suptitle('Final Generated Samples')
plt.tight_layout()
plt.savefig('final_generated_samples.png')
plt.close()

##############################################################
# 추가 과제
#
# 1) nz, ngf, ndf 변경해보기
# 2) Epoch 수 변경해보기
# 3) Monai transform에 Data Augmentation 추가해보기
#
# 다양한 실험을 통해 더 좋은 결과를 얻어보세요!
##############################################################
