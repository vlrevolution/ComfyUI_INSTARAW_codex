from comfy.comfy_types.node_typing import IO

class INSTARAW_SplitByCommas:
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING")
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"
    OUTPUT_NODE = False
    OUTPUT_IS_LIST = [False, False, False, False, False, True]
    DESCRIPTION = "Split the input string into up to five pieces. Splits on commas (or | or ^) and then strips whitespace."

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "string" : ("STRING", {"default":""}), }, "optional": { "split": ([",", "|", "^"], {}), }, }
    
    def func(self, string:str, split:str=",") -> tuple[str,str,str,str,str,list[str]]:
        bits:list[str] = [r.strip() for r in string.split(split)] 
        while len(bits)<5: bits.append("")
        if len(bits)>5: bits = bits[:4] + [",".join(bits[4:]),]
        return (bits[0], bits[1], bits[2], bits[3], bits[4], bits)

class INSTARAW_AnyListToString:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"
    INPUT_IS_LIST  = True
    OUTPUT_IS_LIST = (False,) 

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "anything" : (IO.ANY, ), "join" : ("STRING", {"default":""}), } }
    
    def func(self, anything, join:str):
        return ( join[0].join( [f"{x}" for x in anything] ), )
    
class INSTARAW_StringToInt:
    RETURN_TYPES = ("INT",)
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"

    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "string" : ("STRING", {"default":"", "forceInput":True}), "default" : ("INT", {"default":0}), } }
    
    def func(self, string:str, default:int):
        try: return (int(string.strip()),)
        except: return (default,)

class INSTARAW_StringToFloat:
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "string" : ("STRING", {"default":"", "forceInput":True}), "default" : ("FLOAT", {"default":0}), } }
    
    def func(self, string:str, default:float):
        try: return (float(string.strip()),)
        except: return (default,)

class INSTARAW_ConcatenateStringsNullSafe:
    """
    Concatenates two strings with an optional separator. 
    Crucially, it handles None inputs (from disabled switches) by treating them as empty strings.
    """
    RETURN_TYPES = ("STRING",)
    FUNCTION = "concatenate"
    CATEGORY = "INSTARAW/Utils"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_a": ("STRING", {"default": ""}),
                "string_b": ("STRING", {"default": ""}),
            },
            "optional": {
                "separator": ("STRING", {"default": ", "}),
            }
        }
    
    def concatenate(self, string_a=None, string_b=None, separator=", "):
        # 1. Handle None inputs by converting them to an empty string.
        # This is the "null-safe" part.
        a = string_a if string_a is not None else ""
        b = string_b if string_b is not None else ""
        sep = separator if separator is not None else ""
        
        # 2. Check if both strings are empty after handling None.
        if not a and not b:
            return ("",)
        
        # 3. Concatenate correctly, handling cases where only one string is present
        if not a:
            result = b
        elif not b:
            result = a
        else:
            result = f"{a}{sep}{b}"
            
        return (result,)

class INSTARAW_StringCombine:
    """
    A 'None-safe' string combination node.
    It takes multiple optional strings and a separator, ignores any that are None or empty,
    and joins the valid ones. Perfect for conditional workflows using switches.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {"default": ", ", "multiline": False}),
            },
            "optional": {
                "string_1": ("STRING", {"forceInput": True, "default": ""}),
                "string_2": ("STRING", {"forceInput": True, "default": ""}),
                "string_3": ("STRING", {"forceInput": True, "default": ""}),
                "string_4": ("STRING", {"forceInput": True, "default": ""}),
                "print_to_console": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "combine"
    CATEGORY = "INSTARAW/Utils"

    def combine(self, separator, string_1=None, string_2=None, string_3=None, string_4=None, print_to_console=False):
        # Collect all provided strings into a list
        all_strings = [string_1, string_2, string_3, string_4]
        
        # Filter out any entries that are None or just whitespace
        valid_strings = [s for s in all_strings if s is not None and s.strip()]
        
        # Join the valid strings with the separator
        combined_string = separator.join(valid_strings)
        
        if print_to_console:
            print(f"INSTARAW String Combine Output: '{combined_string}'")
            
        return (combined_string,)