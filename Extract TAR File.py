import tarfile
import os

# Define the path to your tar file
tar_file_path = "C:\\Users\\Administrator\\Desktop\\ML Project\\drkg.tar.gz"  # e.g., "my_data.tar", "project_files.tar.gz"

# Define the directory where you want to extract the contents
extract_path = "C:\\users\\Administrator\\Desktop\\ML Project\\DRKG"

# Create the extraction directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

try:
    # Open the tar file in read mode ('r' for uncompressed, 'r:gz' for gzipped, 'r:bz2' for bzip2)
    with tarfile.open(tar_file_path, "r") as tar:
        # Extract all contents to the specified path
        tar.extractall(path=extract_path)
    print(f"Successfully extracted '{tar_file_path}' to '{extract_path}'.")
except tarfile.ReadError:
    print(f"Error: Could not read tar file '{tar_file_path}'. It might be corrupted or not a valid tar archive.")
except FileNotFoundError:
    print(f"Error: Tar file '{tar_file_path}' not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")