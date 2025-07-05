import pandas as pd

def update_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Assuming the second column is at index 1
    df.iloc[:, 1] = df.iloc[:, 1].map({"good": 1, "bad": 0})

    # Create a new file path with "_updated" appended to the original file name
    new_file_path = file_path.replace(".csv", "_updated.csv")

    # Save the updated DataFrame to the new file
    df.to_csv(new_file_path, index=False)

    print(f"Updated file saved as {new_file_path}")

# Example usage
file_path = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\1Mtech\phishingDetect\phishing_site_urls.csv"
update_csv(file_path)
