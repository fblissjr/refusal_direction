import argparse
import re

def format_prompt_gemma2(prompt):
    parts = re.split(r'<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n', prompt)[1:]
    
    formatted_prompt = "<bos>"
    
    for i in range(0, len(parts), 2):
        role = parts[i]
        content = parts[i+1].strip().replace('<|eot_id|>', '')
        
        if role == 'system':
            continue  # Skip system messages as they're not supported
        elif role == 'assistant':
            role = 'model'
        
        formatted_prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
    
    return formatted_prompt.strip()

def process_file(input_file, output_file, template):
    format_function = {
        'gemma2': format_prompt_gemma2
    }.get(template)
    
    if not format_function:
        raise ValueError(f"Unsupported template: {template}")

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        prompts = infile.read().strip().split('\n')
        for prompt in prompts:
            formatted = format_function(prompt)
            outfile.write(formatted + '\n\n')

def main():
    parser = argparse.ArgumentParser(description="Format prompts according to specified template.")
    parser.add_argument("input_file", help="Path to the input file containing prompts")
    parser.add_argument("output_file", help="Path to the output file for formatted prompts")
    parser.add_argument("--template", choices=['gemma2'], default='gemma2', help="Prompt template to use (default: gemma2)")
    
    args = parser.parse_args()
    
    try:
        process_file(args.input_file, args.output_file, args.template)
        print(f"Formatted prompts have been written to {args.output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
