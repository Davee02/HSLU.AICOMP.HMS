
---
## 1. Init

```
import wandb

# All hyperparameters go into config
config = {
    # Model
    "architecture": "EfficientNet_B0", "pretrained": True,
    # Data
    "fold": 0, "features": "spectrograms", "window_selection": "sum_and_normalize"
    # Training
    "optimizer": "AdamW", "learning_rate": 1e-3, "batch_size": 32, "epochs": 5, "seed": 42, "Scheduler": "CosineAnnealingLR"
}

wandb.init(
    project="HMS-Kaggle-Challenge",
    name="effnetb0-spec-fold0", # Format: [arch]-[features]-[id]
    tags=['baseline', 'fold0'],
    config=config
)
```

---
## 2. Logging Metrics

Use `train/` and `val/` prefixes to group charts.

- **Per Step:** Log learning rate and batch loss.
    Python
    ```
    wandb.log({"train/loss": loss.item(), "train/lr": lr})
    ```
- **Per Epoch:** Log validation metrics. The competition metric is key.
    Python
    ```
    wandb.log({
        "epoch": epoch,
        "val/loss": val_loss,
        "val/kl_div": val_kld_score
    })
    ```
---
## 3. Finish
```
wandb.summary['best_val_kl_div'] = best_kld_score

artifact = wandb.Artifact('model', type='model')
artifact.add_file('path/to/best_model.pth')
wandb.log_artifact(artifact)

wandb.finish()
```
