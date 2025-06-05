import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load CSV ===
file_path = 'Data/Session 1/pass_template_1.csv'
df = pd.read_csv(file_path)

# === Define Sensor Axes ===
orientation_axes = ['orientation_x', 'orientation_y', 'orientation_z']
gyroscope_axes = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']
axes = orientation_axes + gyroscope_axes

# === Function to Plot Sensor Data ===
def plot_data(data, title="Sensor Data"):
    fig, axs = plt.subplots(len(axes), 1, figsize=(12, 14), sharex=True)
    for i, axis in enumerate(axes):
        axs[i].plot(data[axis], label=axis)
        axs[i].set_title(f"{title} - {axis}")
        axs[i].legend()
    plt.tight_layout()
    plt.show()

# === Initial Plot ===
print("\nPlotting full data...")
plot_data(df, title="Full Data")

# === Truncation Loop ===
while True:
    try:
        start = int(input("\nEnter start index for truncation: "))
        end = int(input("Enter end index for truncation: "))
    except ValueError:
        print("Please enter valid integers.")
        continue

    # Validate indices
    if start < 0 or end > len(df) or start >= end:
        print("Invalid indices. Try again.")
        continue

    truncated_df = df.iloc[start:end]
    print("\nPlotting truncated data...")
    plot_data(truncated_df, title=f"Truncated Data [{start}:{end}]")

    confirm = input("Are you happy with the truncation? (y/n): ").strip().lower()
    if confirm == 'y':
        # Export file
        dir_path = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        output_name = f"truncated_{base_name}"
        output_path = os.path.join(dir_path, output_name)

        truncated_df.to_csv(output_path, index=False)
        print(f"\n✅ Truncated CSV saved as:\n{output_path}")
        break
    else:
        print("Okay, let's try different indices.")
