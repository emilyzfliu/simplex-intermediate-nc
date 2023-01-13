import torch
import torchvision.transforms as transforms
from torch import nn

import argparse
from sklearn.neighbors import NearestCentroid
import numpy as np
import math

from tqdm import tqdm
from pathlib import Path
import copy
import pandas as pd
import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torchvision.datasets as datasets

transform = transforms.Compose(
    [transforms.ToTensor(),
   #  transforms.RandomCrop(32),
     #transforms.RandomHorizontalFlip(p=0.5),
   #  transforms.RandomRotation(180),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Normalize((0,), (1.0,)),
     ])
batch_size = 128

class CIFAR10_corrupted(datasets.CIFAR10):
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(CIFAR10_corrupted, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    self.targets = labels


class MLP(nn.Module):
  def __init__(self, L, H, num_etf = 0):
    super().__init__()
    
    self.relu = nn.ReLU()

    layers = [nn.Linear(784, H)]
   #layers = [nn.Linear(1024, H)]
    bns = [nn.BatchNorm1d(H)]

    num_classes = 10

    for i in range(L-2):
      layers.append(nn.Linear(H, H))
      bns.append(nn.BatchNorm1d(H))
        
    # if fixdim:
    #   layers.append(nn.Linear(H, num_classes))
    #   self.fc_out= nn.Linear(num_classes, num_classes)
    #layers.append(nn.Linear(H, H))
    self.fc_out = nn.Linear(H, num_classes)
    
    bns.append(nn.BatchNorm1d(H))

    self.fcs = nn.ModuleList(layers)
    self.bns = nn.ModuleList(bns)

    if num_etf > 0:
      weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
      weight /= torch.sqrt((1/num_classes*torch.norm(weight, 'fro')**2))
      self.fc_out.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, H)))
      self.fc_out.weight.requires_grad_(False)
    
    for i in range(1, num_etf):
      weight = torch.sqrt(torch.tensor(H/(H-1)))*(torch.eye(H)-(1/H)*torch.ones((H, H)))
      weight /= torch.sqrt((1/H*torch.norm(weight, 'fro')**2))
      self.fcs[L-1-i].weight = nn.Parameter(weight)
      self.fcs[L-1-i].weight.requires_grad_(False)

  
  def forward(self, X):
    h = X.reshape(-1, 784)
    for i in range(len(self.fcs)):
      h = self.fcs[i](h)
      h = self.relu(self.bns[i](h))
    h = self.fc_out(h)
    return h, torch.argmax(h, dim=1)

  def get_layer_info(self, X):
    ret = {}
    h = X.reshape(-1, 784)
    for i in range(len(self.fcs)):
      h = self.fcs[i](h)
      ret[str(i+1)] = h
      h = self.relu(self.bns[i](h))
    h = self.fc_out(h)
    ret['fc'] = h
    return ret

  def set_etf(self, key):
    num_classes = 10
    if key == 'fc':
      weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
      weight /= torch.sqrt((1/num_classes*torch.norm(weight, 'fro')**2))
      self.fc_out.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, H)))
      self.fc_out.weight.requires_grad_(False)
    else:
      i = int(key)
      weight = torch.sqrt(torch.tensor(H/(H-1)))*(torch.eye(H)-(1/H)*torch.ones((H, H)))
      weight /= torch.sqrt((1/H*torch.norm(weight, 'fro')**2))
      self.fcs[L-1-i].weight = nn.Parameter(weight)
      self.fcs[L-1-i].weight.requires_grad_(False)


class CONV(nn.Module):
  def __init__(self, L, H, num_etf = 0):
    super().__init__()
    
    self.relu = nn.ReLU()

    starting_layers =  [nn.Conv2d(3, H, kernel_size=2, stride=2), nn.Conv2d(H, H, kernel_size=2, stride=2)]
    starting_bns = [nn.BatchNorm2d(H), nn.BatchNorm2d(H)]

    self.conv_in = nn.ModuleList(starting_layers)
    self.bn_in = nn.ModuleList(starting_bns)

    layers = []
    bns = []

    num_classes = 10

    for i in range(L):
      layers.append(nn.Conv2d(H, H, kernel_size=3, stride=1, padding=1))
      bns.append(nn.BatchNorm2d(H))
        
    self.fc_out = nn.Linear(64*H, 512)
    self.classify = nn.Linear(512, num_classes)

    self.convs = nn.ModuleList(layers)
    self.bns = nn.ModuleList(bns)
    self.softmax = nn.Softmax()

    if num_etf > 0:
      weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
      weight /= torch.sqrt((1/num_classes*torch.norm(weight, 'fro')**2))
      #self.classify.weight = nn.Parameter(weight)
      self.classify.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, 512)))
      self.classify.weight.requires_grad_(False)
    
    for i in range(1, num_etf):
      weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
      weight /= torch.sqrt((1/num_classes*torch.norm(weight, 'fro')**2))
      #self.classify.weight = nn.Parameter(weight)
      self.classify.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, 512)))
      self.classify.weight.requires_grad_(False)
    
  def forward(self, X):
    h = X
    for layer, bn in zip(self.conv_in, self.bn_in):
      h = bn(layer(h))
    h = self.relu(h)

    for i in range(len(self.convs)):
      h = self.convs[i](h)
      h = self.relu(self.bns[i](h))
    h = h.reshape(X.shape[0], -1)
    h = self.fc_out(h)
   # h = self.softmax(self.classify(h))
    return h, torch.argmax(h, dim=1)

  def get_layer_info(self, X):
    ret = {}
    h = X
    for layer, bn in zip(self.conv_in, self.bn_in):
      h = bn(layer(h))
    h = self.relu(h)
    for i in range(len(self.convs)):
      h = self.convs[i](h)
      ret[str(i+1)] = h
      h = self.relu(self.bns[i](h))
    h = h.reshape(X.shape[0], -1)
    h = self.fc_out(h)
    ret['fc'] = h
    return ret


def ncc_accuracy(features, classes):
  clf = NearestCentroid()
  clf.fit(features, classes)
  preds = np.array([clf.predict(features)]).T
  preds = preds.squeeze()
  assert preds.shape == classes.shape
  return np.sum(preds == classes) / preds.shape[0]

def ncc_accuracy_full(feature_means, key, loader, device, model):
  #print(key)
  num_correct = 0.0
  num_seen = 0.0

  bar = tqdm(loader, desc=f'NCC means for layer {key}'.ljust(20))
  for i, batch in enumerate(bar):
    #X: X.shape[0] x 784
    #feature means: 10 x 784
    X, y = batch
    X, y = X.to(device), y.to(device)

    feats = model.get_layer_info(X)[key]
    feats = feats.reshape(1, X.shape[0], -1)
    fmeans = feature_means[key].reshape(1, 10, -1)

    dists = torch.cdist(feats, fmeans)
    dists = dists.reshape(X.shape[0], 10)
    #.reshape(128, 10)
    preds = torch.argmin(dists, dim=1)
    num_correct += torch.sum(preds == y).item()
    num_seen += X.shape[0]
    if i % 10 == 0:
      bar.set_postfix(ncc='{:.2f}'.format(num_correct / num_seen))
  return num_correct / num_seen

def get_evaluations(feature_means, loader, device, model):
  cols = []
  vals = []
  for layer in feature_means:
    cols.append(f'ncc_{layer}')
    vals.append(ncc_accuracy_full(feature_means, layer, loader, device, model))
  return cols, vals

def update_dict(aggregate, batch, labels):
  labels = labels.squeeze()
  if aggregate is None:
    ret = {}
    for elem in batch:
      tot = []
      for i in range(10):
        tot.append(torch.sum(batch[elem][labels == i, :], 0))
      tot = torch.vstack(tot)
      ret[elem] = tot
    return ret
  for elem in aggregate:
    assert elem in batch
    tot = []
    for i in range(10):
      tot.append(torch.sum(batch[elem][labels == i, :], 0))
    tot = torch.vstack(tot)
    aggregate[elem] = aggregate[elem] + tot
  return aggregate

def update_model(model, patience, cols, vals):
  if patience is None:
    patience = {x: 0 for x in cols}
  else:
    for i, key in enumerate(cols):
      if vals[i] > 0.9:
        patience[key] += 1
      else:
        patience[key] = 0
      if patience[key] > 5:
        model.set_etf(key)

def train(model, device, optimizer, scheduler, criterion, dataloaders, savedir, num_epochs, save_interval, adaptive):
  best_loss = np.inf
  losses = []
  model.to(device)
  for epoch in range(num_epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_correct = 0.0
      running_count = 0.0
      num_preds = 0
      bar = tqdm(dataloaders[phase], desc='Epoch {} {}'.format(epoch, phase).ljust(20))

      feature_means = None
      patience = None

      for i, batch in enumerate(bar):
        X, y = batch
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outs, preds = model(X)
          loss = criterion(outs, y)
          outs_layers_batch = model.get_layer_info(X)
          feature_means = update_dict(feature_means, outs_layers_batch, y)
          running_loss += loss.item()
          running_correct+=torch.sum(preds == y).item()
          running_count += preds.shape[0]
          if phase == 'train':
            loss.backward()
            optimizer.step()
        num_preds += 1
        if i % 10 == 0:
          bar.set_postfix(loss='{:.2f}'.format(running_loss / num_preds), acc='{:.2f}'.format(running_correct / running_count))

      if 'phase' == 'train':
        scheduler.step()

      epoch_loss = running_loss / num_preds
      epoch_acc = running_correct / running_count

      for elem in feature_means:
        #print(feature_means[elem].shape)
        feature_means[elem] = feature_means[elem]/running_count

      cols, vals = get_evaluations(feature_means, dataloaders[phase], device, model)
      if adaptive:
        update_model(model, patience, cols, vals)
      df = pd.DataFrame(columns=['epoch', 'phase','loss', 'final_acc']+cols)
      df.loc[0] = [epoch, phase, epoch_loss, epoch_acc] + vals
      losses.append(df)
      if (epoch + 1) % save_interval == 0:
        checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': scheduler}
        Path(os.path.join(savedir, f'checkpoint_epoch_{epoch+1}.pth')).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, os.path.join(os.path.join(savedir, f'checkpoint_epoch_{epoch+1}.pth')))
        losses_int = pd.concat(losses, axis=0, ignore_index=True)
        losses_int.to_csv(os.path.join(savedir, f'losses_epoch_{epoch+1}.csv'), index=False)

  #model.load_state_dict(best_model_wts)
  model.eval()

  # Save model weights
  Path(os.path.join(savedir, 'model.pth')).parent.mkdir(parents=True, exist_ok=True)
  torch.save(model.state_dict(), os.path.join(savedir, 'model.pth'))

  losses = pd.concat(losses, axis=0, ignore_index=True)
  losses.to_csv(os.path.join(savedir, 'losses.csv'), index=False)

  return model


def main(args):
  n_layers = args.n_layers
  etf = args.num_etf
  model = MLP(n_layers, 400, num_etf=etf)
  device = torch.device("cuda:0")
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)
  criterion = torch.nn.CrossEntropyLoss()
  savedir = args.savedir
  adaptive = args.adaptive

  # trainset_10 = CIFAR10_corrupted(root='./data', train=True, corrupt_prob=args.corrupt_percent/100,
  #                                       download=True, transform=transform)

  trainset_10 = datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
  trainloader_10 = torch.utils.data.DataLoader(trainset_10, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

  # valset_10 = CIFAR10_corrupted(root='./data', train=False, corrupt_prob=args.corrupt_percent/100,
  #                                      download=True, transform=transform)
  valset_10 = datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
  valloader_10 = torch.utils.data.DataLoader(valset_10, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
  dataloaders = {
    'train': trainloader_10,
    'val': valloader_10
  }
  save_interval = 50
  num_epochs = 400
  train(model, device, optimizer, scheduler, criterion, dataloaders, savedir, num_epochs, save_interval, adaptive)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="parse args")
  parser.add_argument('-s', '--savedir', default='.', type=str)
  parser.add_argument('-l', '--n-layers', default=8, type=int)
  parser.add_argument('-e', '--num-etf', default=0, type=int)
  parser.add_argument('-c', '--corrupt_percent', default=0, type=float)
  parser.add_argument('-a', '--adaptive', default=False, type=bool)
  args = parser.parse_args()
  main(args)