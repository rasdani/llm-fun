### A100 80GB, 1 CPU, 8GB RAM
```
FIRST RUN:
ELAPSED TIME finetune.py:
end_time - start_time=8.11267638206482
eval_loss=5.7626953125
train_runtime=0.9157

Best by evaluation loss:
{'hyperparams': {'learning_rate': 2e-05, 'per_device_train_batch_size': 1, 'num_train_epochs': 3, 'weight_decay': 0.05, 'gradient_accumulation_steps': 1}, 'eval_loss': 4.734135627746582, 'elapsed_time': 1.5141923427581787, 'train_runtime': 0.9253}

Best by training runtime:
{'hyperparams': {'learning_rate': 2e-05, 'per_device_train_batch_size': 3, 'num_train_epochs': 3, 'weight_decay': 0.01, 'gradient_accumulation_steps': 4}, 'eval_loss': 5.724442481994629, 'elapsed_time': 0.9431865215301514, 'train_runtime': 0.3598}
```
### Colab T4 TPU 15GB, ? CPU, 12.7GB RAM
```
ELAPSED TIME finetune.py:
end_time - start_time=3.546226739883423
eval_loss=5.700565338134766
train_runtime=1.7762

ELAPSED TIME finetune_token_parallel.py:
end_time - start_time=3.941413640975952
eval_loss=5.700565338134766
train_runtime=1.8923


Best by evaluation loss:
{'hyperparams': {'learning_rate': 2e-05, 'per_device_train_batch_size': 1, 'num_train_epochs': 3, 'weight_decay': 0.01, 'gradient_accumulation_steps': 1}, 'eval_loss': 4.758959770202637, 'elapsed_time': 4.356318473815918, 'train_runtime': 3.2501}

Best by training runtime:
{'hyperparams': {'learning_rate': 2e-05, 'per_device_train_batch_size': 3, 'num_train_epochs': 3, 'weight_decay': 0.05, 'gradient_accumulation_steps': 4}, 'eval_loss': 5.724299430847168, 'elapsed_time': 2.3930389881134033, 'train_runtime': 1.2962}
```
