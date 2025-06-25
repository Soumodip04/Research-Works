# ✅ Imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ✅ Load dataset
df = pd.read_csv("air_pollution_data.csv")
df.dropna(inplace=True)

# ✅ Setup features and target
features = ['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3']
target = 'pm2_5'

# ✅ Simulate 3 clients (cities)
cities = df['city'].unique()[:3]
client_datasets = []
scaler = MinMaxScaler()

for city in cities:
    city_df = df[df['city'] == city]
    X = scaler.fit_transform(city_df[features])
    y = city_df[target].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    client_datasets.append((train_ds, test_ds, len(X_train)))  # Save train size too

# ✅ Define simple neural network
class PollutionNet(nn.Module):
    def __init__(self):
        super(PollutionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(len(features), 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

# ✅ Local training function
def local_train(model, train_loader, epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

# ✅ Evaluation function (returns MSE, MAE, R2)
def evaluate(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2

# ✅ Federated Learning Simulation
def federated_learning(num_rounds=5):
    global_model = PollutionNet()

    round_mse = []
    round_mae = []
    round_r2 = []

    for round_num in range(num_rounds):
        local_models = []
        client_sizes = []
        local_metrics = []

        print(f"\n🌍 Round {round_num+1}")

        for client_id, (train_ds, test_ds, train_size) in enumerate(client_datasets):
            local_model = PollutionNet()
            local_model.load_state_dict(global_model.state_dict())

            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=16)

            local_train(local_model, train_loader)
            mse, mae, r2 = evaluate(local_model, test_loader)

            print(f"Client {client_id+1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            local_models.append(local_model.state_dict())
            client_sizes.append(train_size)
            local_metrics.append((mse, mae, r2))

        # Weighted Average for Global Model
        new_state_dict = {}
        total_size = sum(client_sizes)
        for key in global_model.state_dict().keys():
            new_state_dict[key] = sum(local_models[i][key] * client_sizes[i] for i in range(len(local_models))) / total_size
        global_model.load_state_dict(new_state_dict)

        # Weighted Average for Metrics
        avg_mse = sum(local_metrics[i][0] * client_sizes[i] for i in range(len(local_metrics))) / total_size
        avg_mae = sum(local_metrics[i][1] * client_sizes[i] for i in range(len(local_metrics))) / total_size
        avg_r2 = sum(local_metrics[i][2] * client_sizes[i] for i in range(len(local_metrics))) / total_size

        round_mse.append(avg_mse)
        round_mae.append(avg_mae)
        round_r2.append(avg_r2)

        print(f"🌟 Global Model After Round {round_num+1}: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, R²={avg_r2:.4f}")

    # ✅ Plotting
    rounds = list(range(1, num_rounds+1))
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rounds, round_mse, marker='o')
    plt.title('MSE vs Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('MSE')
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(rounds, round_mae, marker='o', color='orange')
    plt.title('MAE vs Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('MAE')
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(rounds, round_r2, marker='o', color='green')
    plt.title('R² vs Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('R² Score')
    plt.grid()

    plt.tight_layout()
    plt.show()

    # ✅ Save the final global model
    torch.save(global_model.state_dict(), "federated_pollution_model.pth")
    print("\n✅ Global model saved as 'federated_pollution_model.pth'.")

# ✅ Run
if __name__ == "__main__":
    federated_learning(num_rounds=5)
