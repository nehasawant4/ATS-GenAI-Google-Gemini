# train_model.py

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf


def fine_tune_model(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    df = df.rename(columns={'Category': 'job_description', 'Resume': 'resume'})  # Rename columns for consistency

    # Create a dummy label (binary classification: 0 or 1)
    df['label'] = 1  # This is just a placeholder. Ideally, you should have real labels for fine-tuning.

    dataset = Dataset.from_pandas(df)

    # Load tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize dataset
    def tokenize_function(example):
        return tokenizer(example['job_description'], example['resume'], truncation=True, padding="max_length",
                         max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

    # Convert to TensorFlow datasets
    features = {x: tokenized_dataset[x] for x in ['input_ids', 'attention_mask', 'token_type_ids']}
    labels = tokenized_dataset['label']

    tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    tf_dataset = tf_dataset.shuffle(len(tokenized_dataset)).batch(16)

    # Define optimizer, loss, and metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Train the model
    model.fit(tf_dataset, epochs=3)

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')


# Train the model
fine_tune_model('Resume Screening.csv')
