import torch
import torch.nn as nn
class BreastCancerNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Dropout(0.2),               # ওভারফিটিং কমায়
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()                   # বাইনারি আউটপুট
        )

    def forward(self, x):
        return self.network(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print( device)

num_features = X_train_tensor.shape[1]
model = BreastCancerNN(num_features).to(device)

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor  = X_test_tensor.to(device)
y_test_tensor  = y_test_tensor.to(device)

print(model)