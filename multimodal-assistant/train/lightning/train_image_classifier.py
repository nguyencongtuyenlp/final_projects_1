import os, random, yaml, argparse
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

class LitImage(pl.LightningModule):
    def __init__(self, backbone="resnet18", num_classes=10, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        model_fn = getattr(models, backbone)
        self.backbone = model_fn(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

def make_dm(batch_size=64, num_workers=2):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.CIFAR10(root=".data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    loader = make_dm(**cfg.get("data", {}))
    mdl = LitImage(**cfg.get("model", {}), lr=cfg.get("optim", {}).get("lr", 3e-4))
    trainer = pl.Trainer(**cfg.get("trainer", {}))
    trainer.fit(mdl, train_dataloaders=loader)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="train/configs/image_classifier.yaml")
    args = ap.parse_args()
    main(args.config)
