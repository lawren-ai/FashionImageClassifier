
import torch
from tqdm import tqdm


# Training Loop
def train_loop(model, dataloader, loss_fn, optimizer, accuracy_fn, device):
  size = len(dataloader)
  train_loss, train_acc = 0, 0
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)
    model.train()
    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss
    acc = accuracy_fn(y, y_pred.argmax(dim=1))
    train_acc += acc

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
  train_loss /= size
  train_acc /= size
  return train_loss, train_acc


# Testing Loop
def test_loop(model, dataloader, loss_fn, accuracy_fn, device):
  size = len(dataloader)
  test_loss, test_acc = 0,0
  model.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      test_pred = model(X)

      loss = loss_fn(test_pred, y)
      test_loss += loss
      acc = accuracy_fn(y, test_pred.argmax(dim=1))
      test_acc += acc

    test_loss /= size
    test_acc /= size
    return test_loss, test_acc

def train(model, train_dataloader, test_dataloader, loss_fn, accuracy_fn, optimizer, epochs):
  # create an empty results dictionary
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}

  # loop through training anf test loops for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_loop(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_loss, test_acc = test_loop(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

    # print out what is happening
    print(
        f"Epoch:  {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.2f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.2f}"
    )

    # Updata result dictionary
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results



