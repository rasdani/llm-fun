from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
from datasets import load_dataset
from itertools import product
import time

from finetune import preprocess_function


def grid_search_finetune(hyperparam_grid):
    results = []
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    my_dataset = load_dataset('json', data_files='processed_data.json')
    train_test_split = my_dataset["train"].train_test_split(test_size=0.2)
    train_set = train_test_split["train"]
    test_set = train_test_split["test"]
    
    tokenized_train_set = train_set.map(preprocess_function, batched=True, remove_columns=my_dataset["train"].column_names)
    tokenized_test_set = test_set.map(preprocess_function, batched=True, remove_columns=my_dataset["train"].column_names)
    
    data_collator = DefaultDataCollator()

    for values in product(*hyperparam_grid.values()):
        start_time = time.time()
        
        current_hyperparams = dict(zip(hyperparam_grid.keys(), values))
        model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        model = model.to('cuda')
        
        training_args = TrainingArguments(
            output_dir="my_awesome_qa_model",
            evaluation_strategy="epoch",
            **current_hyperparams
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_set,
            eval_dataset=tokenized_test_set,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Extract training runtime from trainer logs
        train_runtime = [entry.get("train_runtime") for entry in trainer.state.log_history if "train_runtime" in entry][0]
        
        results.append({
            "hyperparams": current_hyperparams, 
            "eval_loss": eval_loss, 
            "elapsed_time": elapsed_time, 
            "train_runtime": train_runtime
        })
        
    return results

# Define hyperparameters to search over
hyperparam_grid = {
    # 'learning_rate': [2e-5, 3e-5],
    'learning_rate': [2e-5],
    'per_device_train_batch_size': [1, 2, 3, 4],
    'per_device_eval_batch_size': [1],
    # 'num_train_epochs': [3],
    'num_train_epochs': [100],
    'weight_decay': [0.01, 0.05],
    'gradient_accumulation_steps': [1, 2, 3, 4]
}

results = grid_search_finetune(hyperparam_grid)
# for res in results:
#     print(res)

# Sort the results by evaluation loss
sorted_by_loss = sorted(results, key=lambda x: x['eval_loss'])
best_by_loss = sorted_by_loss[0]

# Sort the results by training runtime
sorted_by_runtime = sorted(results, key=lambda x: x['train_runtime'])
best_by_runtime = sorted_by_runtime[0]

print("------------------------------------")
print("Best by evaluation loss:")
print(best_by_loss)

print("\nBest by training runtime:")
print(best_by_runtime)
