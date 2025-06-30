import tensorflow as tf
from tensorflow.keras.models import load_model
from Claude_CNN.py import load_dataset, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model_name = "fox_v5"

base_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/"
model_dir = base_dir + "AI results/Model performance/"
output_dir = os.makedirs(model_dir + 'fine_tuning/', exist_ok=True)

fox_image_dir = base_dir + 'processed/Fox/'
not_fox_image_dir = base_dir + 'processed/Not Fox/preprocessed/'

# Load the pre-trained model
base_model = load_model(model_dir + model_name + f'/models/best_model_{model_name}.h5')

# Load the dataset using your existing function
X, y, class_counts = load_dataset(base_dir, classes)
print(f"Total dataset size: {len(X)} images")

# Split into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# Optional: Freeze some layers if you want to keep their weights fixed
# For example, to freeze the first 10 layers:
for layer in base_model.layers[:5]:
    layer.trainable = False

early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    mode='max', 
    restore_best_weights=True,
    verbose=1
)
checkpoint = ModelCheckpoint(
    os.path.join(models_dir, f'best_fine_tuned_model_{run_name}.h5'),
    monitor='val_accuracy',
    save_best_only=True
)

# Fine-tune the model
print("Starting fine-tuning...")
history = pretrained_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Save the fine-tuned model
fine_tuned_model_path = os.path.join(models_dir, f"fine_tuned_model_{run_name}.h5")
pretrained_model.save(fine_tuned_model_path)
print(f"Saved fine-tuned model to {fine_tuned_model_path}")

# If you also want to use your evaluation functions
from Claude_CNN import evaluate_model, plot_training_history, save_training_summary

# Get the actual classes present in your dataset
unique_classes = np.unique(y)
present_classes = [classes[i] for i in unique_classes]

# Generate plots
plot_training_history(history, plots_dir, run_name)

# Evaluate the model
report = evaluate_model(pretrained_model, X_val, y_val, present_classes, plots_dir, run_name)

# Save training summary
summary = save_training_summary(output_dir, present_classes, class_counts, history, report, run_name)

# Save the fine-tuned model
base_model.save(output_dir + f'fine_tuned_model_{model_name}.h5')