# Define your data directories
train_dir = 'path/to/train'
val_dir = 'path/to/validation'
test_dir = 'path/to/test'

# Train and evaluate the model
model = main(train_dir, val_dir, test_dir)

# Save the model
save_model(model, 'skin_disease_model.h5')