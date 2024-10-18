from typing import Tuple 
import json 
import os 
from PIL import Image 


def get_segmentation_from_bbox(bbox) -> list :
    """Getting the Segmentation coordinate for the rectangular bbox 

    Args:
        bbox (list): bbox coordinates

    Returns:
        segmentation_points(list): rectangular segmentation points
    """
    segmentation_points = [] 
    x, y, width, height = bbox
    x_min,y_min = x,y
    x_max = x + width
    y_max = y + height
    segmentation_points.append([x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min])
    return segmentation_points 

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



categories = [{'supercategory': 'Caption', 'id': 1, 'name': 'Caption'},
 {'supercategory': 'Footnote', 'id': 2, 'name': 'Footnote'},
 {'supercategory': 'Formula', 'id': 3, 'name': 'Formula'},
 {'supercategory': 'List-item', 'id': 4, 'name': 'List-item'},
 {'supercategory': 'Page-footer', 'id': 5, 'name': 'Page-footer'},
 {'supercategory': 'Page-header', 'id': 6, 'name': 'Page-header'},
 {'supercategory': 'Picture', 'id': 7, 'name': 'Picture'},
 {'supercategory': 'Section-header', 'id': 8, 'name': 'Section-header'},
 {'supercategory': 'Table', 'id': 9, 'name': 'Table'},
 {'supercategory': 'Text', 'id': 10, 'name': 'Text'},
 {'supercategory': 'Title', 'id': 11, 'name': 'Title'}, 
 {'supercategory': 'Handwriting', 'id': 12, 'name': 'Handwriting'},
 {'supercategory': 'Stamps', 'id': 13, 'name': 'Stamps'}]

# coco_base = { "info": {},
#               "licenses": [], 
#               "images": [],
#               "annotations": [],
#               "categories": []} 


# coco_base["info"] = {
#     "description": "Financial Documents",
#     "url": "www.fusemachines.com",
#     "version": "1.0",
#     "year": 2024,
#     "contributor": "fusemachines",
#     "date_created": "June,2024"
# }

# coco_base["licenses"].append(
#     {
#         "url": "https://opensource.org/licenses/MIT",
#         "id": 1,
#         "name": "MIT License"
#     }
# )
class_to_id = {
    'Caption': 1,
    'Footnote': 2,
    'Formula': 3,
    'List-item': 4,
    'Page-footer': 5,
    'Page-footer': 5, 
    'Page-header': 6,
    'Picture': 7,
    'Section-header': 8,
    'Table': 9,
    'Text': 10,
    'Title': 11,
    'Handwriting': 12,
    'Stamps': 13
}




def create_coco_dataset_from_prediction(img_path:str, json_path: str, output_path:str) -> None : 
    """Create the COCO Dataset with the predictions from the dit 

    Args:
        
        img_path (str): Path of the image 
        json_path (str): Path of the dit predictions 
        output_path (str): Path for the Output COCO JSON 
    """
    coco_base = { 
              "images": [],
              "annotations": [],
              "categories": []} 
    
    coco_base["categories"].extend(categories)
    
    data , name = read_json(json_path)
    # Open the image and get its dimensions
    with Image.open(os.path.join(img_path, name)) as img:
        width, height = img.size 
    coco_base['images'].append( {
                "id": 1,
                "width": width,
                "height": height,
                "file_name": name,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": "2024-06-25"
            }) 
    # Create annotations
    # Change this according to the dit result from the pipeline 
    for idx, (box, category, score) in enumerate(zip(data["dit"]["boxes"], data["dit"]["classes"], data["dit"]["scores"])):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        bbox =  [x_min, y_min, width, height]
        coco_base["annotations"].append({
            "id": idx + 1,
            "image_id": 1,
            "category_id": class_to_id[category],
            "segmentation": get_segmentation_from_bbox(bbox),
            "area": width * height,
            "iscrowd": 0,
            "score": score
        })
    with open(output_path+f"/{name}.json", 'w') as fptr:
        json.dump(coco_base, fptr, indent=4) 
        