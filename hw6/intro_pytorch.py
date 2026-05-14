import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    # Define the transform
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # Load the dataset
    if training:
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    else:
        dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=training)
    return dataloader 


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        # 1. A Flatten layer to convert the 2D pixel array to a 1D array.
        nn.Flatten(),
        # 2. A Dense layer with 128 nodes and a Leaky ReLU activation (negative slope = 0.01)
        nn.Linear(in_features=28*28, out_features=128),
        nn.LeakyReLU(negative_slope=0.01),
        # 3. A Dense layer with 64 nodes and a Leaky ReLU activation (negative slope = 0.01).
        nn.Linear(in_features=128, out_features=64),
        nn.LeakyReLU(negative_slope=0.01),
        # 4. A Dense layer with 10 nodes.
        nn.Linear(in_features=64, out_features=10)
    )
    return model


def build_deeper_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        # 1. A Flatten layer to convert the 2D pixel array to a 1D array.
        nn.Flatten(),
        # 2. A Dense layer with 256 nodes and a LeakyReLU activation
        nn.Linear(28*28, 256), 
        nn.LeakyReLU(0.01),
        # 3. A Dense layer with 128 nodes and a LeakyReLU activation
        nn.Linear(256, 128), 
        nn.LeakyReLU(0.01),
        # 4. A Dense layer with 64 nodes and a LeakyReLU activation
        nn.Linear(128, 64), 
        nn.LeakyReLU(0.01), 
        # 5. A Dense layer with 32 nodes and a LeakyReLU activation
        nn.Linear(64, 32), 
        nn.LeakyReLU(0.01),
        # 6. A Dense layer with 10 nodes.
        nn.Linear(32, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(T):
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            #Update total_loss and correct predictions count
            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total_samples
        avg_loss = total_loss / total_samples
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total_samples}({accuracy:.2f}%) Loss: {avg_loss:.3f}")


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total_samples
    avg_loss = total_loss / total_samples
    
    if show_loss:
        print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    model.eval()
    class_names = [
        'T-shirt/top','Trouser','Pullover','Dress','Coat',
        'Sandal','Shirt','Sneaker','Bag','Ankle Boot'
    ]
    with torch.no_grad():
        image = test_images[index].unsqueeze(0)
        logits = model(image)

        probs = F.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probs, 3)

        for i in range(3):
            label = class_names[top_indices[0][i].item()]
            probability = top_probs[0][i].item() * 100
            print(f"{label}: {probability:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
