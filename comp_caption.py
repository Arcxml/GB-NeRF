import os
import argparse

def main(folder_path, prefix):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print("Folder does not exist, please check the path.")
        return

    # Get all files in the folder and sort them
    files = sorted(os.listdir(folder_path))

    # Iterate over all files and modify their content
    for filename in files:
        if filename.endswith('.txt') and os.path.isfile(os.path.join(folder_path, filename)):
            # Read the original content
            old_file_path = os.path.join(folder_path, filename)
            with open(old_file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Add the prefix to the content
            new_content = f"{prefix}, {content}"

            # Write the new content back to the file
            with open(old_file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)

    print("All files have been updated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a prefix to the content of all txt files in a specified folder")
    parser.add_argument('--folder', type=str, required=True, help='Folder path')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix to add to the content')
    args = parser.parse_args()
    
    main(args.folder, args.prefix)