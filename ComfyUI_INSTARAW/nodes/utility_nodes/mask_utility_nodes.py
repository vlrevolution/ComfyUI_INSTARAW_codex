import torch

class INSTARAW_MaskedSection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "image": ("IMAGE",),
                "minimum": ("INT", {"default":512, "min":16, "max":4096})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "func"
    CATEGORY = "INSTARAW/Utils"
    
    def func(self, mask:torch.Tensor, image, minimum=512):
        mbb = mask.squeeze()
        H,W = mbb.shape
        masked = mbb > 0.5

        non_zero_positions = torch.nonzero(masked)
        if len(non_zero_positions) < 2: return (image,)

        min_x = int(torch.min(non_zero_positions[:, 1]))
        max_x = int(torch.max(non_zero_positions[:, 1]))
        min_y = int(torch.min(non_zero_positions[:, 0]))
        max_y = int(torch.max(non_zero_positions[:, 0]))

        if (x:=(minimum-(max_x-min_x))//2)>0:
            min_x = max(min_x-x, 0)
            max_x = min(max_x+x, W)
        if (y:=(minimum-(max_y-min_y))//2)>0:
            min_y = max(min_y-y, 0)
            max_y = min(max_y+y, H)       

        return (image[:,min_y:max_y,min_x:max_x,:],)


class INSTARAW_MaskCombine:
    """
    Combines two masks using various mathematical operations.
    If only one mask is provided, it will be passed through unchanged.
    """
    
    OPERATION_MODES = ["Union (Max)", "Add (Clamped)", "Subtract (Clamped)", "Intersection (Min)", "Average"]

    @classmethod
    def INPUT_TYPES(cls):
        # --- THIS IS THE FIX: Made masks optional ---
        return {
            "required": {
                "operation": (cls.OPERATION_MODES, {"default": "Union (Max)"}),
            },
            "optional": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine_masks"
    CATEGORY = "INSTARAW/Utils"

    def combine_masks(self, operation, mask1=None, mask2=None):
        # --- THIS IS THE FIX: Added passthrough logic for single masks ---
        if mask1 is None and mask2 is None:
            # If both are missing, we can't create a mask of unknown size. Return None.
            return (None,)
        if mask1 is None:
            print("INSTARAW Mask Combine: mask1 is missing, passing through mask2.")
            return (mask2,)
        if mask2 is None:
            print("INSTARAW Mask Combine: mask2 is missing, passing through mask1.")
            return (mask1,)
        # --- END FIX ---

        # If we get here, both masks are present. Proceed with combination logic.
        target_device = mask1.device
        mask2 = mask2.to(target_device)

        if mask1.shape != mask2.shape:
            if mask1.numel() > mask2.numel():
                target_shape = mask1.shape[1:]
                mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            else:
                target_shape = mask2.shape[1:]
                mask1 = torch.nn.functional.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=target_shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        if operation == "Union (Max)":
            result = torch.max(mask1, mask2)
        elif operation == "Add (Clamped)":
            result = torch.clamp(mask1 + mask2, 0.0, 1.0)
        elif operation == "Subtract (Clamped)":
            result = torch.clamp(mask1 - mask2, 0.0, 1.0)
        elif operation == "Intersection (Min)":
            result = torch.min(mask1, mask2)
        elif operation == "Average":
            result = (mask1 + mask2) / 2.0
        else:
            result = torch.max(mask1, mask2)

        return (result,)