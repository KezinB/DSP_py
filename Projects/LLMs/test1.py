# Step 1: Install required libraries
# Run this in your terminal or notebook:
# pip install torch transformers datasets

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Step 2: Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or other variants for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 3: Load and preprocess a C++ dataset
# For this demo, we'll use a small dataset of C++ code snippets
# You can replace this with a larger dataset like CodeSearchNet
def load_cpp_dataset():
    # Example: A small dataset of C++ code snippets
    cpp_code_snippets = [
        "int add(int a, int b) { return a + b; }",
        "int subtract(int a, int b) { return a - b; }",
        "int multiply(int a, int b) { return a * b; }",
        "int divide(int a, int b) { return a / b; }",
        "void printHello() { std::cout << \"Hello, World!\"; }",
        "int factorial(int n) { return (n <= 1) ? 1 : n * factorial(n - 1); }",
        "int fibonacci(int n) { return (n <= 1) ? n : fibonacci(n - 1) + fibonacci(n - 2); }",
    ]
    return cpp_code_snippets

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=64)

# Prepare the dataset
cpp_code_snippets = load_cpp_dataset()
tokenized_datasets = tokenize_function(cpp_code_snippets)

# Convert to PyTorch dataset
class CppDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

dataset = CppDataset(tokenized_datasets)

# Step 4: Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save the model
    evaluation_strategy="epoch",    # Evaluate every epoch
    learning_rate=2e-5,             # Learning rate
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,   # Batch size for evaluation
    num_train_epochs=3,             # Number of training epochs
    weight_decay=0.01,              # Weight decay
    save_steps=10_000,              # Save model every 10,000 steps
    save_total_limit=2,             # Limit the number of saved models
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

# Fine-tune the model
trainer.train()

# Step 5: Generate C++ code
def generate_cpp_code(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_code

# Example usage
prompt = "// Write a C++ function to calculate the factorial of a number\n"
generated_code = generate_cpp_code(prompt)
print("Generated C++ Code:\n", generated_code)