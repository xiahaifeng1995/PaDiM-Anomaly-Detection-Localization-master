import torch
import os
import random
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor
from scipy.spatial.distance import mahalanobis
from PIL import Image
from tqdm import tqdm
import time
import numpy as np
from random import sample
import pickle

random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)

outputs = []

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x

def plot_fig(img, im_name, scores, threshold, save_dir, class_name):
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    #  img = np.array(test_img)
    heat_map = scores * 255
    mask = scores
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask *= 255
    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
    fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
    fig_img.subplots_adjust(right=0.9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)
    ax_img[0].imshow(img)
    ax_img[0].title.set_text('Image')
    #  ax_img[1].imshow(gt, cmap='gray')
    #  ax_img[1].title.set_text('GroundTruth')

    ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
    ax_img[1].imshow(img, cmap='gray', interpolation='none')
    ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
    ax_img[1].title.set_text('Predicted heat map')

    ax_img[2].imshow(mask, cmap='gray')
    ax_img[2].title.set_text('Predicted mask')
    ax_img[3].imshow(vis_img)
    ax_img[3].title.set_text('Segmentation result')
    left = 0.92
    bottom = 0.15
    width = 0.015
    height = 1 - 2 * bottom
    rect = [left, bottom, width, height]
    cbar_ax = fig_img.add_axes(rect)
    cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 8,
    }
    cb.set_label('Anomaly Score', fontdict=font)

    fig_img.savefig(os.path.join(save_dir, f'{im_name}'), dpi=100)
    plt.close()

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).cuda()
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def hook(module, input, output):
    outputs.append(output)

t_d = 1792
d = 550
threshold=0.3
class_name='iphone'
idx = torch.tensor(sample(range(0, t_d), d)).cuda()
dataset_root='/home/acer/iphone/train/good'
save_path='/home/shared/samba/s2/s3/test/padim'
test_dataset_root='/home/acer/iphone/test/bad'
train_feature_filepath='/home/acer/embeddings.npy'

device='cuda:0'
model= AutoModel.from_pretrained('Ramos-Ramos/dino-resnet-50').cuda()
model.eval()
feature_extractor = AutoFeatureExtractor.from_pretrained('Ramos-Ramos/dino-resnet-50')
if not os.path.exists(train_feature_filepath):
    feat1=[]
    feat2=[]
    feat3=[]
    for im_name in tqdm(os.listdir(dataset_root)):
        image=Image.open(os.path.join(dataset_root, im_name))
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs=model(**inputs, output_hidden_states=True)
        feat1.append(outputs.hidden_states[1])
        feat2.append(outputs.hidden_states[2])
        feat3.append(outputs.hidden_states[3])
    feat1=torch.cat(feat1, dim=0)
    feat2=torch.cat(feat2, dim=0)
    feat3=torch.cat(feat3, dim=0)
    embedding_vectors=embedding_concat(feat1, feat2)
    embedding_vectors=embedding_concat(embedding_vectors, feat3)

    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0).cpu().numpy()

    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False) + 0.01 * I
    # save learned distribution
    print ('save train set feature to: %s' % train_feature_filepath)
    train_outputs = [mean, cov]
    with open(train_feature_filepath, 'wb') as f:
        pickle.dump(train_outputs, f)
    print ('save train set feature to: %s done' % train_feature_filepath)
else:
    # load train_feature_filepath from: train_feature_filepath
    print('load train set feature from: %s' % train_feature_filepath)
    with open(train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)
    print('the mean of train outputs shape is %s' % str(train_outputs[0].shape))
    print('the cov of train outputs shape is %s' % str(train_outputs[1].shape))

    mean=torch.from_numpy(train_outputs[0]).cuda()
    cov = torch.from_numpy(train_outputs[1]).cuda()

    cov_inv= []
    #  B, C, H, W = embedding_vectors.size()

    print('compute conv_inv')
    s=time.time()
    for i in range(cov.size(-1)):
        cov_inv_i = torch.inverse(cov[:, :, i])
        cov_inv.append(cov_inv_i)
    print(f'process conv_inv takes {time.time()-s} seconds')

    for im_name in tqdm(os.listdir(test_dataset_root)):
        print('process test image: %s' % im_name)
        image=Image.open(os.path.join(test_dataset_root, im_name))
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        input_shape=inputs['pixel_values'].shape
        # for visualizaiton only
        denorm_img=denormalization(inputs['pixel_values'].squeeze().cpu().numpy())
        with torch.no_grad():
            outputs=model(**inputs, output_hidden_states=True)

        feat1=outputs.hidden_states[1]
        feat2=outputs.hidden_states[2]
        feat3=outputs.hidden_states[3]
        embedding_vectors=embedding_concat(feat1, feat2)
        embedding_vectors=embedding_concat(embedding_vectors, feat3)

        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        #  dist_list = []
        dist_list = torch.zeros(size=(H*W, B))
        s=time.time()
        # cpu model is slow
        #  for i in range(H * W):
            #  mean = train_outputs[0][:, i]
            #  conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            #  dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            #  dist_list.append(dist)

        for i in range(H*W):
            delta = embedding_vectors[:, :, i] - mean[:, i]
            m_dist = torch.sqrt(torch.diag(torch.mm(torch.mm(delta, cov_inv[i]), delta.t())))
            dist_list[i] = m_dist

        dist_list = dist_list.transpose(1, 0).view(B, H, W)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=input_shape[2], mode='bilinear', align_corners=False).squeeze().cpu().numpy()

        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        print('prcossing time takes %s seconds' % str(time.time()-s))

        #  dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        #  dist_list = torch.tensor(dist_list)
        #  score_map = F.interpolate(dist_list.unsqueeze(1), size=input_shape[2], mode='bilinear',
                                  #  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        #  for i in range(score_map.shape[0]):
            #  score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        plot_fig(denorm_img, im_name, scores, threshold, save_path, class_name)
