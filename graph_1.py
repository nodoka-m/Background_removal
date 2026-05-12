import matplotlib.pyplot as plt

history = {
    "loss": [3.1402, 2.6122, 2.3090, 2.1105, 1.8632, 1.6967, 1.6812, 1.4732],
    "accuracy": [0.150, 0.306, 0.516, 0.597, 0.416, 0.613, 0.616, 0.697],        # ← val_acc
    "val_accuracy": [0.166, 0.259, 0.478, 0.519, 0.475, 0.541, 0.603, 0.662]   # ← test_acc
}

metrics = ['loss', 'accuracy']

plt.figure(figsize=(10, 5))

for i in range(len(metrics)):
    metric = metrics[i]

    plt.subplot(1, 2, i+1)
    plt.title(metric)

    plt_train = history[metric]
    plt.plot(plt_train, label='training')

    if 'val_' + metric in history:
        plt_val = history['val_' + metric]
        plt.plot(plt_val, label='validation')

    plt.legend()

plt.tight_layout()
plt.show()