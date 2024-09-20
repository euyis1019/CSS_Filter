import os

# Specify the directory path to traverse
input_directory = input("Input Absolute Path, end with PascalVOC12/splits:")

# Traverse all files in the folder
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):  # Only process .txt files
        input_file_path = os.path.join(input_directory, filename)

        # Open the input file and read content
        with open(input_file_path, 'r') as infile:
            lines = infile.readlines()

        # List to store processed lines
        converted_lines = []

        # Process each line and prepare for writing
        for line in lines:
            # Strip leading and trailing whitespace, and replace the first '/' with ''
            cleaned_line = line.strip().replace('/', '', 1)  # Replace the first '/' with empty
            cleaned_line = cleaned_line.replace(' /', ',')  # Replace space with comma
            converted_lines.append(cleaned_line)  # Add the processed line to the list

        # Overwrite the original file with updated content
        with open(input_file_path, 'w') as outfile:
            outfile.write('\n'.join(converted_lines) + '\n')  # Write the new lines

        print(f"File updated: {filename}")

print("All files processed successfully!")
