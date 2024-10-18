import os 
from typing import Tuple 
import json 
import logging 
from COCOUtils.convert2coco import create_coco_dataset_from_prediction 
from COCOUtils.coco2ls import convert_coco_to_ls
import argparse

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')




def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse directories for processing.")
    parser.add_argument('--img_dir', type=str, required=True, help='Directory for images')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory for JSON files from pipeline')
    parser.add_argument('--coco_dir', type=str, required=True, help='Directory for Saving COCO annotations')
    parser.add_argument('--ls_dir', type=str, required=True, help='Directory for saving label-studio output')

    # Parsing the arguments
    args = parser.parse_args()
    return args 

def read_json (json_path: str ) -> Tuple : 
    """reads the json 

    Args:
        json_path (str): path of the json file 

    Returns:
        Tuple: Tupe of Dict and string (name of the file )
    """
    with open(json_path,'r') as fptr : 
        json_file = json.load(fptr)
    file_name = f"{json_path.split('/')[-1].split('.json')[0]}"
    return json_file, file_name

def run_pipeline(IMG_DIR, JSON_DIR,COCO_DIR, LS_DIR): 

    for json_file in os.listdir(JSON_DIR):   
        full_json_file_path =  os.path.join(JSON_DIR, json_file)
        _, name = read_json(full_json_file_path)
        try: 
            create_coco_dataset_from_prediction(IMG_DIR, full_json_file_path, COCO_DIR)
            full_coco_dir = os.path.join(COCO_DIR, f"{name}.json")

            full_ls_dir = os.path.join(LS_DIR, f"{name}.json")
            convert_coco_to_ls(full_coco_dir, full_ls_dir)
        except FileNotFoundError:
            logging.info(f"Missing, name: {name}")


if __name__=="__main__": 
    args = parse_arguments() 
    IMG_DIR = args.img_dir 
    JSON_DIR = args.json_dir 
    COCO_DIR = args.coco_dir 
    LS_DIR = args.ls_dir     
    run_pipeline(IMG_DIR, JSON_DIR, COCO_DIR, LS_DIR)




