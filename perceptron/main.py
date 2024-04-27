import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron


# Preparing the training data
df = pd.read_csv("perceptron_toydata.csv", sep="\t")
x_train = df[["x1", "x2"]].values
y_train = df["label"].values


# Loading a pre-trained Perceptron model
# ppn = Perceptron.load_model("perceptron_model.h5")


# Training a new Perceptron model
ppn = Perceptron(num_features=2, learning_rate=0.9)

ppn.train(x_train, y_train, epochs=10, verbose=True) # Training the perceptron with training data

train_acc = ppn.compute_accuracy(x_train, y_train)
print(f"\nTraining Accuracy: {train_acc*100}%")


# Saving the model
ppn.save_model("perceptron_model.h5")


# Making predictions
test_data = [1.4, 2.5]
prediction = ppn.predict(test_data)
print(f"Prediction: {prediction}")


# Plotting the decision boundary
x1_min, x1_max, x2_min, x2_max = ppn.decision_boundary()

## Plotting class 0
plt.plot(
    x_train[y_train == 0, 0],
    x_train[y_train == 0, 1],
    marker="D",
    markersize=10,
    linestyle="",
    label="Class 0",
)

## Plotting class 1
plt.plot(
    x_train[y_train == 1, 0],
    x_train[y_train == 1, 1],
    marker="^",
    markersize=13,
    linestyle="",
    label="Class 1",
)

## Plotting the decision boundary
plt.plot([x1_min, x1_max], [x2_min, x2_max], color="k")

plt.legend(loc=2)

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.xlabel("Feature $x_1$", fontsize=12)
plt.ylabel("Feature $x_2$", fontsize=12)

plt.grid()
plt.show()