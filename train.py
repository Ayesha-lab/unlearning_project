from Models.mlp import MLP
import mlflow
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def train_model(model_state, train_loader, lr, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    with mlflow.start_run() as run:
        mlflow.log_params({"lr": lr, "epochs": epochs})

        model = MLP(input_dim=784, output_dim=10)  # Create fresh model
        model.load_state_dict(model_state)
        model = model.to(device)  # Move model to GPU
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # mlflow context and experiment
        # itertools loop
        for epoch in range(epochs):
            print(f"epoch: {epoch}")
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.view(X_batch.size(0), -1)
                X_batch = X_batch.to(device)  # Move batch to GPU
                y_batch = y_batch.to(device)  # Move labels to GPU

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", accuracy, step=epoch)

    return model