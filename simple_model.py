import torch
from torch import nn
from torch import optim
from GithubDatasetExtraction import get_article
from block_roberta import block_based_embedding

class Classifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_classes=3, num_layers=10):
        super().__init__()

        layers = []
        # Input Layer
        layers.append(nn.Linear(input_size, hidden_size))

        # Hidden Layers (ELU)
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ELU())

        # Output Layer (Logit + softmax)
        layers.append(nn.Linear(hidden_size, num_classes))
        layers.append(nn.Softmax(dim=1))

        # Add layers together.
        self.network = nn.Sequential(*layers)

        # SEt loss function.
        self.criterion = nn.NLLLoss()
        self.optimizer = None


    def forward(self, article):
        return self.network(article)


    def predict(self, input_data):
        self.eval()
        with torch.no_grad():
            # Ensure input is a 2D tensor [1, 768]
            x = torch.tensor(input_data, dtype=torch.float32).view(1, -1)
            probs = self.forward(x)
            confidence, predicted = torch.max(probs, 1)
            return {
                "class": predicted.item(),
                "confidence": confidence.item()
            }


    def train_model(self, train_data, labels, epochs=10, lr=1e-4, epoch_log = 1):
        """
        Trains the model using the given training data and labels.\n
        Training data must be organized into a list of 768 long embbeding vectors and the labels must be converted to an
        int value between 0 and 2 (inclusive).
        :param epoch_log:
        :param train_data:
        :param labels:
        :param epochs:
        :param lr:
        :return:
        """
        self.train()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        inputs = torch.tensor(train_data, dtype=torch.float32)
        targets = torch.tensor(labels, dtype=torch.long)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output_probs = self.forward(inputs)

            loss = self.criterion(torch.log(output_probs + 1e-9), targets)

            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % epoch_log == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')


    def save_model(self, file_path):
        """Saves the model weights to a file."""
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")


    def load_model(self, file_path):
        """Loads the model weights from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()
        print(f"Model loaded from {file_path}")
