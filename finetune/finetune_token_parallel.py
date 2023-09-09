import time
start_time = time.time()
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import DefaultDataCollator
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        # print(f"{sequence_ids}")

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs



my_dataset = load_dataset('json', data_files='processed_data.json')
train_test_split = my_dataset["train"].train_test_split(test_size=0.2)  # 80% train, 20% test
train_set = train_test_split["train"]
test_set = train_test_split["test"]

# tokenized_data = my_dataset.map(preprocess_function, batched=True, remove_columns=my_dataset["train"].column_names)
tokenized_train_set = train_set.map(preprocess_function, batched=True, remove_columns=my_dataset["train"].column_names, num_proc=4)
tokenized_test_set = test_set.map(preprocess_function, batched=True, remove_columns=my_dataset["train"].column_names, num_proc=4)


data_collator = DefaultDataCollator()


model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

model = model.to('cuda')
print(f"{model.device=}")


training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
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


print("ELAPSED TIME finetune_token_parallel.py:")
print(f"{end_time - start_time=}")
print(f"{eval_loss=}")
for entry in trainer.state.log_history:
    if "train_runtime" in entry:
        train_runtime = entry["train_runtime"]
        print(f"{train_runtime=}")

# trainer.push_to_hub()
