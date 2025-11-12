# Filename: ComfyUI_INSTARAW/nodes/utility_nodes/list_utility_nodes.py

import torch

# We no longer need IO from comfy_types
# from comfy.comfy_types.node_typing import IO

class INSTARAW_BatchFromImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "images": ("IMAGE", ), } }
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"

    def func(self, images):
        if not images:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        if len(images) == 1:
            return (images[0],)
        else:
            return (torch.cat(list(i for i in images), dim=0),)
        
class INSTARAW_ImageListFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "images": ("IMAGE", ), } }
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = [True,]
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"

    def func(self, images):
        image_list = list( i.unsqueeze(0) for i in images )
        return (image_list,) 
    
class INSTARAW_StringListFromStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "s0": ("STRING", {"default":""}), 
                "s1": ("STRING", {"default":""}), 
            },
            "optional": {    
                "s2": ("STRING", {"default":""}), 
                "s3": ("STRING", {"default":""}), 
            }
        }
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = [True,]
    RETURN_TYPES = ("STRING", )
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"

    def func(self, s0,s1,s2=None,s3=None):
        lst = [s0,s1]
        if s2 is not None: lst.append(s2)
        if s3 is not None: lst.append(s3)
        return (lst,) 

class INSTARAW_PickFromList:
    @classmethod
    def INPUT_TYPES(cls):
        # --- THE DEFINITIVE FIX ---
        # Replace the non-serializable IO.ANY type object with the string "*"
        return { "required": {"anything" : ("*", ), "indexes": ("STRING", {"default": "0"})} }
        # --- END FIX ---

    RETURN_TYPES = ("*",) # Use "*" for output too for consistency
    RETURN_NAMES = ("picks",)
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = [True,]

    def func(self, anything, indexes):
        if not anything:
            return ([],)
            
        try:
            # The input list can sometimes be nested, e.g. [[item1, item2]]
            if len(anything) == 1 and isinstance(anything[0], list):
                anything = anything[0]
            
            # The 'indexes' input is a single string from the widget, not a list
            indexes_str = indexes if isinstance(indexes, str) else str(indexes)
            parsed_indexes = [int(x.strip()) for x in indexes_str.split(',') if x.strip()]
        except Exception as e:
            print(f"INSTARAW PickFromList: Error parsing indexes '{indexes}'. Defaulting to first item. Error: {e}")
            parsed_indexes = [0]
        
        valid_picks = []
        for i in parsed_indexes:
            if 0 <= i < len(anything):
                valid_picks.append(anything[i])
            else:
                print(f"INSTARAW PickFromList Warning: index {i} is out of bounds for list of size {len(anything)}. Skipping.")
        
        return (valid_picks, )