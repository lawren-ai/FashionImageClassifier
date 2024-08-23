
import torch 
import data_setup, engine, model_builder, utils

from torchvision import transforms

# setup hyperoarameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 16
LEARNING_RATE = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    transform=transforms.ToTensor(),
    batch_size=BATCH_SIZE
)

# Create model instance from model builder
model = model_builder.MNISTModel(
    input_shape=784,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)


# set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training 
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model
utils.save_model(model=model,
                 target_dir = "models",
                 model_name = "fashion_classifier_model.pth"
                 )

