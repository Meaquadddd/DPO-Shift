import os
import json
import argparse
from urllib import response

def extract_responses(input_dir, output_file):
    """
    Reads all JSON files in the specified directory, extracts the embedded JSON from the 'response' field,
    and saves them into a combined JSON file.

    :param input_dir: Path to the directory containing JSON files.
    :param output_file: Path to the output JSON file.
    """
    combined_responses = []
    if not os.path.exists(output_file):
            # 使用open函数以写入模式打开文件，若文件不存在会自动创建，这里只是创建了空文件，若要写入内容可在open后进行相应操作
        with open(output_file, 'w') as f:
            pass
    else:
        print(f"{output_file} 文件已存在,自动清空")
        open(output_file, 'w').close()
    # Iterate over all files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            
            try:
                # Open and load the JSON file
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    # Extract and parse the 'response' field
                    if 'response' in data:
                        # response_json = json.loads(data['response'])
                        response_json = {}
                        response_json['question_id'] = data['question_id']
                        response_json['response'] = data['response']
                        combined_responses.append(response_json)
                        with open(output_file, 'a') as f:
                            f.write(json.dumps(response_json) + '\n')
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error processing file {filename}: {e}")
    
    # # Write the combined responses to the output file
    # with open(output_file, 'w') as output_json:
    #     json.dump(combined_responses, output_json, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embedded JSON from response fields in JSON files.")
    parser.add_argument("--input-dir", required=True, help="Path to the directory containing JSON files.")
    parser.add_argument("--output-file", required=True, help="Path to the output JSON file.")
    
    args = parser.parse_args()
    
    extract_responses(args.input_dir, args.output_file)
