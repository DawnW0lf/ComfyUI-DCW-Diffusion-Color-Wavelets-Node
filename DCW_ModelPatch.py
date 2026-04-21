import torch
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

                # --- DYNAMIC BROADCASTING FIX ---
                # We count how many dimensions the latent has (4 for image, 5 for video)
                # and expand the scale 's' to match.
                dims = len(input_x.shape)
                s = timestep.view(-1, *([1] * (dims - 1)))
                
                if s.shape[0] != output.shape[0]:
                    s = s[0].repeat(output.shape[0], *([1] * (dims - 1)))

                # If 5D (Video), we squeeze the 'Frames' dim to do Wavelet math, then unsqueeze
                is_5d = (dims == 5)
                if is_5d:
                    # [B, C, F, H, W] -> [B*F, C, H, W]
                    b, c_feat, f, h, w = input_x.shape
                    work_input = input_x.transpose(1, 2).reshape(-1, c_feat, h, w)
                    work_output = output.transpose(1, 2).reshape(-1, c_feat, h, w)
                    work_s = s.transpose(1, 2).reshape(-1, 1, 1, 1)
                else:
                    work_input = input_x
                    work_output = output
                    work_s = s

                # 1. Calculate x0
                x0 = work_input - (work_s * work_output)
                
                # 2. Wavelet Logic
                yl_t, yh_t = dwt(work_input)
                yl_0, yh_0 = dwt(x0)
                
                t_val = timestep[0].item()
                t_factor = min(t_val / 1000.0, 1.0) if t_val > 1.0 else t_val
                
                yl_corrected = yl_0 + (strength * (yl_t - yl_0) * t_factor)
                x0_corrected = idwt((yl_corrected, yh_0))
                
                # 3. Final Prediction
                res_output = (work_input - x0_corrected) / torch.clamp(work_s, min=1e-7)

                # 4. Restore 5D shape if necessary
                if is_5d:
                    # [B*F, C, H, W] -> [B, C, F, H, W]
                    res_output = res_output.view(b, f, c_feat, h, w).transpose(1, 2)

                if t_val % 10 == 0:
                    print(f"[DCW] Success | Mode: {'5D' if is_5d else '4D'} | T: {t_val:.2f}")

                return res_output

            except Exception as e:
                print(f"[DCW ERROR] Still failing: {e}")
                return output

        m = model.clone()
        m.model_options["model_function_wrapper"] = dcw_wrapper
        return (m,)

NODE_CLASS_MAPPINGS = {"DCW_ModelPatch": DCW_ModelPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"DCW_ModelPatch": "Apply DCW (Wavelet Patch)"}