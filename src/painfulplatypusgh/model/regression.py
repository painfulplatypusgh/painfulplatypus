import torch
import matplotlib.pyplot as plt
import numpy as np

class LinearRegression:
    #Constructor
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs

        #Initialize w0 and w1
        self.w0 = torch.randn(1, requires_grad=True, dtype=torch.float32)
        self.w1 = torch.randn(1, requires_grad=True, dtype=torch.float32)

        #Loss function: MSE
        self.loss = torch.nn.MSELoss()

        #Optimizer: SGD
        self.optimizer = torch.optim.SGD([self.w0, self.w1], lr=self.lr)

        #Tracking lists
        self.history_w0 = []
        self.history_w1 = []
        self.history_loss = []

    #Forward function
    def forward(self, x):
        #Computes y = w1 * x + w0
        return self.w1 * x + self.w0

    #Fit function
    def fit(self, X_train, y_train, X_test, y_test):

        #Make sure inputs are 1d tensors
        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        X_te = torch.tensor(X_test, dtype=torch.float32)
        y_te = torch.tensor(y_test, dtype=torch.float32)

        #Training loop
        for epoch in range(self.epochs):

            #Clear out old gradient values
            self.optimizer.zero_grad()

            #Generate current predictions
            y_pred = self.forward(X_tr)

            #Compute loss
            loss = self.loss(y_pred, y_tr)

            #Run backpropagation
            loss.backward()

            #Update parameters
            self.optimizer.step()

            self.history_w0.append(self.w0.item())
            self.history_w1.append(self.w1.item())
            self.history_loss.append(loss.item())

        #Evaluate on test set
        with torch.no_grad():
            test_preds = self.forward(X_te)
            ss_res = torch.sum((y_te - test_preds)**2)
            ss_tot = torch.sum((y_te - torch.mean(y_te))**2)
            r2 = 1 - (ss_res/ss_tot)

        print(f"Test Set R2 Score =  {r2.item():.6f}")

    #Predict function
    def predict(self, bcr_data):
        #Predicts Annual Production given BCR data
        with torch.no_grad():
            bcr_tensor = torch.tensor(bcr_data, dtype=torch.float32)
            annual_production = self.forward(bcr_tensor)
        return annual_production.numpy()

    def analysis_plot(self, X_data, y_data):
        fig, axs = plt.subplots(2, 2)

        #Original data plot
        X_arr = np.array(X_data)
        y_arr = np.array(y_data)
        X_line = np.linspace(X_arr.min(), X_arr.max(), 100)
        y_line = self.w1.item() * X_line + self.w0.item()

        axs[0, 0].scatter(X_arr, y_arr, color='blue', label='Original Data')
        axs[0, 0].plot(X_line, y_line, color='red', label='Fitted Line')
        axs[0, 0].set_title('Original Data and Fitted Regression Line')
        axs[0, 0].set_xlabel('BCR')
        axs[0, 0].set_ylabel('Annual Production')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        #Training Loss Progression
        axs[0, 1].plot(self.history_loss, color='purple')
        axs[0, 1].set_title('Loss Progression')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Loss (MSE)')
        axs[0, 1].grid(True)




        #Intercept (w0) progression
        axs[1, 0].plot(self.history_w0, color='green')
        axs[1, 0].set_title('Intercept (w0) Progression')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Value of w0')
        axs[1, 0].grid(True)

        #Slope (w1) progression
        axs[1, 1].plot(self.history_w1, color='orange')
        axs[1, 1].set_title('Slope (w1) Progression')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Value of w1')
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()