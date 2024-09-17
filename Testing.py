import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
def read_and_plot(csv_file, x_column, y_column):
    try:
        # Load the CSV into a pandas DataFrame
        data = pd.read_csv(csv_file)

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(data[x_column], data[y_column], marker='o')

        # Add labels and title
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'{y_column} vs {x_column}')

        # Display the plot
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"File '{csv_file}' not found.")
    except KeyError:
        print(f"Columns '{x_column}' or '{y_column}' not found in the CSV file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage: specify the CSV file and columns to plot
    csv_file = 'STAT2150_mid.csv'  # Replace with your actual file
    x_column = 'year'         # Replace with your actual X-axis column
    y_column = 'class'         # Replace with your actual Y-axis column

    read_and_plot(csv_file, x_column, y_column)
