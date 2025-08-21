import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = "results/resnet18_cifar10_binary_hinge_test/log.txt"

# Regex pattern to capture the summary lines per epoch
pattern = re.compile(
    r"Epoch:\s*(\d+).*?Training Prec@1\s*([\d.]+).*?Training Prec@5\s*([\d.]+).*?"
    r"Validation Prec@1\s*([\d.]+).*?Validation Prec@5\s*([\d.]+)"
)

# Lists to store values
epochs, train_prec1, train_prec5, val_prec1, val_prec5 = [], [], [], [], []

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_prec1.append(float(match.group(2)))
            train_prec5.append(float(match.group(3)))
            val_prec1.append(float(match.group(4)))
            val_prec5.append(float(match.group(5)))

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_prec1, marker='o')
plt.title("Train Prec@1 per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Prec@1")

plt.subplot(2, 2, 2)
plt.plot(epochs, train_prec5, marker='o', color='orange')
plt.title("Train Prec@5 per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Prec@5")

plt.subplot(2, 2, 3)
plt.plot(epochs, val_prec1, marker='o', color='green')
plt.title("Validation Prec@1 per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Prec@1")

plt.subplot(2, 2, 4)
plt.plot(epochs, val_prec5, marker='o', color='red')
plt.title("Validation Prec@5 per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Prec@5")

plt.tight_layout()
plt.show()
