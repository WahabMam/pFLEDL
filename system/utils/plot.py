import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
csv_path = '/home/khoa/Khoa/PFLLib/system/results/Cifar10-partition_dir-clients_20-niid_True-balance_False-batch_10-classpc_10-alpha_01-train_075_FedEDL_test_0.csv'  # Change to your CSV file location
df = pd.read_csv(csv_path)

 

rounds = range(len(df))

# Calculate gap: accuracy - ECE (for personalized FL only)
gap_per = df['acc_per'] - df['ece_per']

# Plot accuracy and ECE with dual y-axis
fig, ax1 = plt.subplots(figsize=(8,5))

color_acc = 'tab:blue'
color_ece = 'tab:red'

ax1.set_xlabel('Round')
ax1.set_ylabel('Accuracy', color=color_acc)
ax1.plot(rounds, df['acc_per'], color=color_acc, marker='o', label='Personalized Accuracy')
ax1.tick_params(axis='y', labelcolor=color_acc)
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.set_ylabel('ECE', color=color_ece)
ax2.plot(rounds, df['ece_per'], color=color_ece, marker='s', label='Personalized ECE')
ax2.tick_params(axis='y', labelcolor=color_ece)
ax2.set_ylim(0, 1)

fig.tight_layout()
plt.title('Personalized Accuracy and ECE per Round')
plt.show()

# Plot accuracy - ECE gap
plt.figure(figsize=(8,5))
plt.plot(rounds, gap_per, label='Accuracy - ECE Gap', marker='d', color='green')
plt.xlabel('Round')
plt.ylabel('Accuracy - ECE')
plt.title('Gap Between Accuracy and Calibration Error (Personalized FL)')
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('personalized_accuracy_ece_gap.png')
print("Saved plot: personalized_accuracy_ece_gap.png")

# Print final round summary
print(f"Final Round Personalized Accuracy: {df['acc_per'].iloc[-1]:.4f}")
print(f"Final Round Personalized ECE: {df['ece_per'].iloc[-1]:.4f}")
print(f"Final Round Gap (Accuracy - ECE): {gap_per.iloc[-1]:.4f}")

