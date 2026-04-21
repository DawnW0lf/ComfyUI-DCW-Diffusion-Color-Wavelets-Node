import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class DCW_ModelPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "wavelet": (["haar", "db2", "db3"], {"default": "haar"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_dcw"
    CATEGORY = "model_patches/DCW"

    def apply_dcw(self, model, strength, wavelet):
        dwt = DWTForward(J=1, wave=wavelet, mode='zero')
        idwt = DWTInverse(wave=wavelet, mode='zero')

        def dcw_wrapper(model_function, params):
            input_x = params.get("input") 
            timestep = params.get("timestep")
            c = params.get("c")
            
            output = model_function(input_x, timestep, **c)
            
            if strength == 0:
                return output

            try:
                device, dtype = output.device, output.dtype
                dwt.to(device=device, dtype=dtype)
                idwt.to(device=device, dtype=dtype)

                # 1. Determine Dimensions
                dims = len(input_x.shape)
                # Correctly broadcast S to [B, 1, 1, 1] or [B, 1, 1, 1, 1]
                s = timestep.view(-1, *([1] * (dims - 1)))
                
                if s.shape[0] != output.shape[0]:
                    s = s[0].repeat(output.shape[0], *([1] * (dims - 1)))

                # 2. 5D to 4D Conversion for Wavelets
                is_5d = (dims == 5)
                if is_5d:
                    b, c_feat, f, h, w = input_x.shape
                    work_input = input_x.transpose(1, 2).reshape(-1, c_feat, h, w)
                    work_output = output.transpose(1, 2).reshape(-1, c_feat, h, w)
                    work_s = s.transpose(1, 2).reshape(-1, 1, 1, 1)
                else:
                    work_input = input_x
                    work_output = output
                    work_s = s

                # 3. Handle Padding
                orig_h, orig_w = work_input.shape[-2:]
                pad_h = orig_h % 2
                pad_w = orig_w % 2
                
                if pad_h > 0 or pad_w > 0:
                    work_input = F.pad(work_input, (0, pad_w, 0, pad_h))
                    work_output = F.pad(work_output, (0, pad_w, 0, pad_h))
                    # Ensure work_s matches the padded spatial dimensions
                    work_s_pad = work_s.expand(-1, -1, work_input.shape[-2], work_input.shape[-1])
                else:
                    work_s_pad = work_s

                # 4. Calculate x0
                # If it fails here, the debug prints above will show why
                x0 = work_input - (work_s_pad * work_output)
                
                # 5. Wavelet Correction
                yl_t, yh_t = dwt(work_input)
                yl_0, yh_0 = dwt(x0)
                
                t_val = timestep[0].item()
                t_factor = min(t_val / 1000.0, 1.0) if t_val > 1.0 else t_val
                
                yl_corrected = yl_0 + (strength * (yl_t - yl_0) * t_factor)
                x0_corrected = idwt((yl_corrected, yh_0))
                
                # 6. Crop and Return
                if pad_h > 0 or pad_w > 0:
                    x0_corrected = x0_corrected[:, :, :orig_h, :orig_w]
                    work_input = work_input[:, :, :orig_h, :orig_w]

                res_output = (work_input - x0_corrected) / torch.clamp(work_s, min=1e-7)

                if is_5d:
                    res_output = res_output.view(b, f, c_feat, orig_h, orig_w).transpose(1, 2)

                return res_output

            except Exception as e:
                print(f"\n[DCW FATAL ERROR]: {e}")
                try:
                    print(f"Context Shapes -> Input: {work_input.shape}, Output: {work_output.shape}, S: {work_s_pad.shape}")
                except:
                    pass
                return output

        m = model.clone()
        m.model_options["model_function_wrapper"] = dcw_wrapper
        return (m,)

NODE_CLASS_MAPPINGS = {"DCW_ModelPatch": DCW_ModelPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"DCW_ModelPatch": "Apply DCW (Wavelet Patch)"}
