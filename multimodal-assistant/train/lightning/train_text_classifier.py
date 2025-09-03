import yaml, argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, DataCollatorWithPadding

class LitText(pl.LightningModule):
    def __init__(self, hf_model="distilbert-base-uncased", num_labels=2, lr=3e-5):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model, num_labels=num_labels)
        self.lr = lr
        self.train_ds = None
        self.collator = DataCollatorWithPadding(self.tokenizer)

    def prepare_data(self):
        raw = load_dataset("glue", "sst2")
        tok = self.tokenizer
        def proc(ex):
            out = tok(ex["sentence"], truncation=True)
            out["labels"] = ex["label"]
            return out
        self.train_ds = raw["train"].map(proc, batched=True, remove_columns=raw["train"].column_names)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=16, shuffle=True, collate_fn=self.collator)

    def training_step(self, batch, batch_idx):
        out = self.model(**{k: v for k, v in batch.items()})
        loss = out.loss
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    mdl = LitText(**cfg.get("model", {}), lr=cfg.get("optim", {}).get("lr", 3e-5))
    trainer = pl.Trainer(**cfg.get("trainer", {}))
    trainer.fit(mdl)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="train/configs/text_classifier.yaml")
    args = ap.parse_args()
    main(args.config)
