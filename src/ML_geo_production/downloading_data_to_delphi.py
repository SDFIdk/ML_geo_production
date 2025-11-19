import os
import subprocess
import argparse
import time

def main(remote_dir, output_dir, files, number_of_loops):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the list of files to download
    with open(files, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    for loop in range(number_of_loops):
        print(f"Starting download loop {loop + 1}/{number_of_loops}")
        for filename in file_list:
            # Construct the remote URL and local file path
            remote_url = f"{remote_dir}/{filename}"
            local_path = os.path.join(output_dir, filename)
            
            # Check if the file already exists
            if os.path.exists(local_path) and not os.path.exists(local_path + ".part"):
                print(f"File '{filename}' already downloaded. Skipping.")
                continue
            
            # Download the file with wget, continue if partially downloaded
            try:
                print(f"Downloading '{filename}' from {remote_url}")
                subprocess.run(
                    ["wget", "-c","-P", output_dir, remote_url],
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to download '{filename}', moving to next file.")
        
        print(f"Completed loop {loop + 1}/{number_of_loops}\n")
        time.sleep(1)  # Optional: Add a short pause between loops if needed

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Download files with wget in multiple loops.")
    parser.add_argument("--remote_dir", default="https://test33.dataforsyningen.dk/computerome/fast_trueorto/", help="Remote directory containing files (default: %(default)s)")
    parser.add_argument("--output_dir", default="./output", help="Local directory to save downloaded files (default: %(default)s)")
    parser.add_argument("--files", default="./files.txt", help="Text file listing filenames to download (default: %(default)s)")
    parser.add_argument("--number_of_loops", type=int, default=3, help="Number of times to repeat the download process (default: %(default)s)")
    
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(args.remote_dir, args.output_dir, args.files, args.number_of_loops)
