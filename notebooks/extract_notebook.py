import json
import os
import argparse

def extract_notebook_to_txt(notebook_path, output_txt_path):
    # Verify the notebook exists
    if not os.path.exists(notebook_path):
        print(f"Error: Could not find '{notebook_path}'")
        return

    # Load the notebook JSON data
    with open(notebook_path, 'r', encoding='utf-8') as f:
        try:
            nb_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: The file is not a valid Jupyter Notebook (JSON).")
            return

    print(f"Reading '{notebook_path}'...")

    # Open the target text file for writing
    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
        cells = nb_data.get('cells', [])
        
        for i, cell in enumerate(cells, 1):
            cell_type = cell.get('cell_type', 'unknown')
            source_lines = cell.get('source', [])
            source_text = "".join(source_lines)
            
            # Write Cell Header (e.g., Cell 1 [CODE] or Cell 2 [MARKDOWN])
            out_file.write(f"{'='*60}\n")
            out_file.write(f"Cell {i} [{cell_type.upper()}]\n")
            out_file.write(f"{'='*60}\n")
            
            # Write the actual code or markdown content
            out_file.write(source_text.strip() + "\n\n")
            
            # If it's a code cell, process the outputs
            if cell_type == 'code':
                outputs = cell.get('outputs', [])
                if outputs:
                    out_file.write(f"--- OUTPUT ---\n")
                    for output in outputs:
                        output_type = output.get('output_type', '')
                        
                        # Handle standard print statements
                        if output_type == 'stream':
                            text = "".join(output.get('text', []))
                            out_file.write(text)
                            
                        # Handle execution results (like returning a dataframe or variable)
                        elif output_type in ('execute_result', 'display_data'):
                            data = output.get('data', {})
                            # Grab plain text output if it exists
                            if 'text/plain' in data:
                                text = "".join(data['text/plain'])
                                out_file.write(text + "\n")
                                
                        # Handle errors directly in the output
                        elif output_type == 'error':
                            ename = output.get('ename', 'Error')
                            evalue = output.get('evalue', '')
                            out_file.write(f"[ERROR: {ename}] {evalue}\n")
                    out_file.write("\n\n")
                else:
                    out_file.write("--- NO OUTPUT ---\n\n\n")
            else:
                out_file.write("\n\n")

    print(f"Success! Notebook contents saved to '{output_txt_path}'")

if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Extract Jupyter Notebook to a text file.")
    parser.add_argument("input", help="Path to the .ipynb file")
    # get txt name from ipynb file name
    parser.add_argument("-o", "--output", default=None, 
                        help="Path to the output .txt file (default: notebook_context.txt)")
    
    args = parser.parse_args()
    
    # Set default output name if not provided
    if args.output is None:
        args.output = args.input.replace(".ipynb", ".txt")
    
    extract_notebook_to_txt(args.input, args.output)