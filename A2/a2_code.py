import torch
from torch import nn, optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRAINING_SIZE = 15000
VALIDATION_SIZE = 1000


def load_dataset():
    # Define transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,)),
                                          ])
    dataset = datasets.ImageFolder(root='notMNIST_small')
    total_size = len(dataset.targets)
    train_indices, val_indices = train_test_split(list(
        range(total_size)), test_size=total_size - TRAINING_SIZE, stratify=dataset.targets)

    # Use train_test_split twice to split dataset into train, validation and test sets.
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    val_indices, test_indices = train_test_split(val_indices, test_size=total_size - TRAINING_SIZE - VALIDATION_SIZE, stratify=[
                                                 dataset.targets[i] for i in range(len(dataset.targets)) if i in val_indices])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Set transformation
    train_dataset.dataset = copy(train_dataset.dataset)
    val_dataset.dataset = copy(val_dataset.dataset)
    test_dataset.dataset = copy(test_dataset.dataset)

    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform
    test_dataset.dataset.transform = transform

    print(train_dataset.dataset.transform,
          val_dataset.dataset.transform, test_dataset.dataset.transform)
    # Load dataset with batch size = 32
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=True)
    return trainloader, validationloader, testloader


def define_model(hidden_sizes: int):
    # define the neural network model
    # Hyperparameters for our network
    input_size = 3 * 28 * 28
    output_size = 10
    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes, output_size),
                          nn.Softmax(dim=1))
    print(model)
    return model


def define_model_two_layers(hidden_sizes: list):
    # define the neural network model
    # Hyperparameters for our network
    input_size = 3 * 28 * 28
    output_size = 10
    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.Softmax(dim=1))
    print(model)
    return model


def define_model_with_dropout(hidden_sizes: int):
    # define the neural network model
    # Hyperparameters for our network
    input_size = 3 * 28 * 28
    output_size = 10
    # Build a feed-forward network
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes),
                          nn.ReLU(),
                          nn.Dropout(0.5),
                          nn.Linear(hidden_sizes, output_size),
                          nn.Softmax(dim=1))
    print(model)
    return model


def train_model(model, trainloader, validationloader, save_file_name, lr, n_epochs=20, max_epochs_stop=2):
    # create model and define the loss

    criterion = nn.CrossEntropyLoss()
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr)
    overall_start = timer()
    history = []
    valid_loss_min = np.Inf

    for epoch in range(n_epochs):
        train_loss = 0
        validation_loss = 0

        train_acc = 0
        validation_acc = 0

        model.train()
        start = timer()

        for ii, (images, labels) in enumerate(trainloader):
            # Flatten MNIST images into a 2352 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * images.size(0)
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(trainloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')
        else:
            with torch.no_grad():
                model.eval()

                for images, labels in validationloader:
                    images = images.view(images.shape[0], -1)
                    # forward pass
                    output = model(images)

                    # validation loss
                    loss = criterion(output, labels)
                    validation_loss += loss.item() * images.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(labels.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    validation_acc += accuracy.item() * images.size(0)

                # calculate average loss and average accuracy for training and validation
                train_loss = train_loss / len(trainloader.dataset)
                validation_loss = validation_loss / \
                    len(validationloader.dataset)

                train_acc = train_acc / len(trainloader.dataset)
                validation_acc = validation_acc / len(validationloader.dataset)

                history.append(
                    [train_loss, validation_loss, train_acc, validation_acc])

                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {validation_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * validation_acc:.2f}%'
                )
                if validation_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = validation_loss
                    best_epoch = epoch

                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f}'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history, valid_loss_min

    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f}'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history, valid_loss_min


def test_error_acc(model, testloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_acc = 0
    for images, labels in testloader:
        images = images.view(images.shape[0], -1)
        # forward pass
        output = model(images)

        # test loss
        loss = criterion(output, labels)
        test_loss += loss.item() * images.size(0)

        # Calculate test accuracy
        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(labels.data.view_as(pred))
        accuracy = torch.mean(
            correct_tensor.type(torch.FloatTensor))
        # Multiply average accuracy times the number of examples
        test_acc += accuracy.item() * images.size(0)
    test_loss /= len(testloader.dataset)
    test_acc /= len(testloader.dataset)
    return test_loss, test_acc


if __name__ == '__main__':
    HIDDEN_SIZE = 1000
    FILE_NAME = 'nn_model.pt'

    trainloader, validationloader, testloader = load_dataset()
    print(len(trainloader.dataset), len(
        validationloader.dataset), len(testloader.dataset))

    # ============= Task 2 ==============
    LR_RATE = [0.1, 0.05, 0.03, 0.01, 0.005]
    valid_loss_min = np.Inf
    for lr in LR_RATE:
        model = define_model(HIDDEN_SIZE)
        print(f'Trianing on leanring rate: {lr}')
        model, history, valid_loss = train_model(
            model, trainloader, validationloader, FILE_NAME, lr, n_epochs=20)
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            best_model = model
            best_history = history
            best_lr = lr
    print('The best learning rate is ', best_lr)
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for c in ['train_loss', 'valid_loss']:
        ax1.plot(best_history[c], label=c)
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross Entropy loss')
    ax1.set_title('Training and Validation losses')

    for c in ['train_acc', 'valid_acc']:
        ax2.plot(best_history[c], label=c)
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average accuracy')
    ax2.set_title('Training and Validation accuracy')

    test_loss, test_acc = test_error_acc(best_model, testloader)
    print(
        f'The test loss is {test_loss}. The test accuracy is {100 * test_acc:.2f}%')
    plt.show()

    # =========== task 3 ================
    # Based on task 2, the best learning rate is 0.1
    try:
        lr = best_lr
    except:
        lr = 0.1
    valid_loss_min = np.Inf
    layer_size = [100, 500, 1000]
    valid_loss_list = []
    best_size = 0
    for size in layer_size:
        model = define_model(size)
        print(f"Training for size: {size}")
        model, history, valid_loss = train_model(
            model, trainloader, validationloader, FILE_NAME, lr, n_epochs=20)
        valid_loss_list.append(valid_loss)
        if valid_loss < valid_loss_min:
            best_size = size
            valid_loss_min = valid_loss
            best_model = model
    print(
        f'The validation loss are {valid_loss_list[0]:.4f}, {valid_loss_list[1]:.4f}, {valid_loss_list[2]:.4f}')
    test_loss, test_acc = test_error_acc(best_model, testloader)
    print(
        f'The best model has {best_size} units in hidden layer')
    print(
        f'The test loss is {test_loss}. The test accuracy is {100 * test_acc:.2f}%')

    # =========== task 4 ================
    try:
        lr = best_lr
    except:
        lr = 0.1
    model = define_model_two_layers([500, 500])
    model, history, valid_loss = train_model(
        model, trainloader, validationloader, FILE_NAME, lr, n_epochs=20)
    try:
        fig.clear()
    except:
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    ax3 = fig.add_subplot(121)
    ax4 = fig.add_subplot(122)
    for c in ['train_loss', 'valid_loss']:
        ax3.plot(history[c], label=c)
    ax3.legend()
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Cross Entropy loss')
    ax3.set_title('Training and Validation losses')

    for c in ['train_acc', 'valid_acc']:
        ax4.plot(history[c], label=c)
    ax4.legend()
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Average accuracy')
    ax4.set_title('Training and Validation accuracy')

    test_loss, test_acc = test_error_acc(model, testloader)
    print(
        f'The test loss is {test_loss}. The test accuracy is {100 * test_acc:.2f}%')
    plt.show()

    # =========== task 5 ================
    try:
        lr = best_lr
    except:
        lr = 0.1
    model = define_model_with_dropout(HIDDEN_SIZE)
    model, history, valid_loss = train_model(
        model, trainloader, validationloader, FILE_NAME, lr, n_epochs=20)
    try:
        fig.clear()
    except:
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    ax5 = fig.add_subplot(121)
    ax6 = fig.add_subplot(122)
    for c in ['train_loss', 'valid_loss']:
        ax5.plot(history[c], label=c)
    ax5.legend()
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Cross Entropy loss')
    ax5.set_title('Training and Validation losses')
    for c in ['train_acc', 'valid_acc']:
        ax6.plot(history[c], label=c)
    ax6.legend()
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Average accuracy')
    ax6.set_title('Training and Validation accuracy')

    test_loss, test_acc = test_error_acc(model, testloader)
    print(
        f'The test loss is {test_loss}. The test accuracy is {100 * test_acc:.2f}%')
    plt.show()
