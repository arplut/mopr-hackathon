# Connecting Google Colab to Weights & Biases

A complete guide to set up your ML workflow: Code on GitHub → Develop in Google Colab → Track Experiments with Weights & Biases

---

## Step 1: Install W&B in Colab

```python
!pip install wandb
```

---

## Step 2: Authenticate with W&B

### Option A: Interactive Login (Recommended)

```python
import wandb

# This will prompt you to login
wandb.login()
```

When you run this, Colab will give you a link. Click it, log in to W&B, copy your API key, and paste it back in Colab.

### Option B: Direct API Key

```python
import wandb
wandb.login(key='YOUR_WANDB_API_KEY')
```

Get your API key from: https://wandb.ai/settings/keys

---

## Step 3: Initialize a W&B Project

```python
import wandb

# Start a new experiment run
wandb.init(
    project="mopr-hackathon",  # Your project name
    entity="your-username",     # Your W&B username
    name="experiment-1"         # Optional: name this run
)
```

---

## Step 4: Log Metrics During Training

```python
# In your training loop
for epoch in range(10):
    loss = train_one_epoch()
    accuracy = evaluate()
    
    # Log to W&B
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy
    })

# Finish the run
wandb.finish()
```

---

## Complete Colab Workflow Example

Here's what your Colab notebook should look like:

### Cell 1: Clone GitHub repo

```python
!git clone https://github.com/your-username/mopr-hackathon.git
import os
os.chdir('/content/mopr-hackathon')
```

### Cell 2: Install dependencies

```python
!pip install wandb torch  # Add other libs you need
```

### Cell 3: Login to W&B

```python
import wandb
wandb.login()
```

### Cell 4: Import your training code

```python
from train import train_model
```

### Cell 5: Initialize W&B and train

```python
wandb.init(
    project="mopr-hackathon",
    entity="your-username",
    name="run-1"
)

# Train and log metrics
results = train_model(log_to_wandb=True)

wandb.finish()
```

### Cell 6: (Optional) Push code changes back to GitHub

```python
!git add .
!git commit -m "Training run from Colab"
!git push
```

---

## Modify Your Training Script

If your `train.py` has a training loop, modify it to log to W&B:

```python
import wandb

def train_model(log_to_wandb=True):
    model = create_model()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = train_step(batch)
            
            if log_to_wandb:
                wandb.log({"loss": loss, "epoch": epoch})
        
        accuracy = evaluate()
        if log_to_wandb:
            wandb.log({"accuracy": accuracy, "epoch": epoch})
    
    return model
```

---

## What Gets Logged to W&B

You can log:

- **Metrics**: loss, accuracy, F1-score, etc.
- **Hyperparameters**: learning rate, batch size, model architecture
- **Plots**: confusion matrices, sample predictions
- **Model checkpoints**: save best model to W&B
- **Tables**: predictions, errors, dataset samples

### Log Hyperparameters

```python
wandb.config.update({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
})
```

### Log Plots

```python
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=labels, 
        preds=predictions
    )
})
```

### Save Model

```python
wandb.save('best_model.pth')
```

---

## Quick Command Reference

```python
import wandb

wandb.login()                                    # Authenticate
wandb.init(project="...", entity="...")         # Start run
wandb.log({"metric": value})                    # Log metric
wandb.config.update({"param": value})           # Log hyperparameter
wandb.finish()                                  # End run
```

---

## View Results

After training:

1. Go to https://wandb.ai
2. Click your project
3. View all runs, compare metrics, see charts
4. Share reports with your team

---

## Your Complete ML Workflow

```
Code Repository (GitHub)
    ↓
Development Environment (Google Colab)
    ↓
Experiment Tracking (Weights & Biases)
    ↓
Results & Analysis
```

- **Code** → GitHub (version control)
- **Develop/Train** → Google Colab (free GPU)
- **Track Experiments** → Weights & Biases (metrics, visualization, collaboration)

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'wandb'`

Make sure you installed it:
```python
!pip install wandb
```

### Authentication fails

Try logging in again with a fresh API key:
```python
wandb.login()
```

### Metrics not appearing in W&B

Check that:
1. `wandb.init()` is called before logging
2. `wandb.log()` is called with valid data
3. `wandb.finish()` is called at the end (or let it auto-finish)

---

## Next Steps

- Explore W&B [Reports](https://docs.wandb.ai/guides/reports) to visualize results
- Use [Artifacts](https://docs.wandb.ai/guides/artifacts) to version models and datasets
- Set up [Alerts](https://docs.wandb.ai/guides/alerts) for important metrics
- [Share reports](https://docs.wandb.ai/guides/reports/save-reports) with your team
