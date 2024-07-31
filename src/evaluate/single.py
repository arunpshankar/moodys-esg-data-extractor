from src.utils.evaluate import compare_json_objects
from src.config.logging import logger 
from src.config.setup import config
from src.utils.io import load_jsonl
import json
import os



def compare_jsonl_files(file1_path, file2_path):
    json_list1 = load_jsonl(file1_path)
    json_list2 = load_jsonl(file2_path)
    
    matches = []
    for obj2 in json_list2:
        for obj1 in json_list1:
            is_code_matched, is_value_matched   = compare_json_objects(obj1, obj2)
            if is_code_matched and is_value_matched:
                matches.append((obj1, obj2))
                
    return matches, len(json_list2)


if __name__ == '__main__':
    file_name = '100395060535523152'
    workflow = 'single_step'

    ground_truth_file_path = os.path.join(config.DATA_DIR, f'validation/expected/{file_name}.jsonl')
    extracted_data_file_path = os.path.join(config.DATA_DIR, f'validation/generated/{workflow}/{file_name}.jsonl')
    x, y = compare_jsonl_files(extracted_data_file_path, ground_truth_file_path, )
    print()
    for item in x:
        a, b = item 
        print(a)
        print()
        print(b)
        print('-' * 100)
    print(y)
    print(len(x))



