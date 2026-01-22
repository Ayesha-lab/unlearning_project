from Models.mlp import MLP
import torch
from data_splitter import MNISTLoader
from train import train_model
from Unlearning.certified_unlearning import CertifiedUnlearning
import itertools
from mlflow_setup import setup_mlflow
import mlflow
import webbrowser
import time


"""" 
Create and load initial model: model_I and save for reuse
create dataset and split it into train and test set
further split train set into retain and forget set
train model_I on train set -> model_T
train model_I on retain set -> model_R
unlearn forget set on model_T -> model_U
"""

def main():
    start_time = time.time()
    ui_url = setup_mlflow()
    model_i = MLP(input_dim = 784, output_dim = 10)
    torch.save(model_i, 'model_i.pth')
    params = {'lr': [0.1, 0.05, 0.01],
              'epochs': [15, 30, 60],
              'batch_size': [512],
              }
    for lr, epochs, batch_size in itertools.product(*params.values()):

        with mlflow.start_run(run_name=f"lr_{lr}_epochs_{epochs}_bs_{batch_size}"):
            mlflow.log_params({
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "epsilon": 1.0,
                "delta": 1e-5,
                "num_iterations": 500
            })

        loader = MNISTLoader(data_dir='./mnist_data', train_split=0.7, retain_split=0.5, batch_size= batch_size)
        train_loader, test_loader, retain_loader, forget_loader = loader.load_data()
        print("beginning training")
        model_t = train_model(model_i.state_dict(), train_loader, lr, epochs)
        print(f"training model on full dataset, time: {time.time() - start_time}")
        model_r = train_model(model_i.state_dict(), retain_loader, lr, epochs)
        print(f"training model on retain dataset, time: {time.time() - start_time}")


        torch.save(model_t, f'model_t_{lr}_{epochs}_{batch_size}.pth')
        torch.save(model_r, f'model_r_{lr}_{epochs}_{batch_size}.pth')

        unlearner = CertifiedUnlearning(model_t.state_dict())

        model_u = unlearner.unlearn(
            retain_loader=retain_loader,
            epsilon=1.0,
            delta=1e-5,
            num_iterations=500,
            learning_rate=lr,
            clip_norm_0=1.0,
            clip_norm_1=1.0,
            verbose=True
        )
        torch.save(model_u.state_dict(), f'model_u_{lr}_{epochs}_{batch_size}.pth')

        model_ft  = unlearner.post_unlearn_finetune(
            retain_loader=retain_loader,
            num_epochs=50,
            learning_rate=lr,
            weight_decay=0.0005,
            verbose=True
        )

        torch.save(model_ft.state_dict(), f'model_ft_{lr}_{epochs}_{batch_size}.pth')

if __name__ == "__main__":
    main()



