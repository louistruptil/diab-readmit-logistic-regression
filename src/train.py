import torch
import torch.nn as nn
import torch.optim as optim
from model import LogisticRegression
from preprocess import X_train, Y_train

model = LogisticRegression(8)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for epoch in range(1, epochs + 1) :
    y_pred = model(X_train)
    loss = criterion(y_pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(loss.item())