import os
import dill 

base_path = "/home/vvikash/jax_privacy/pretrained/tpu_diffusion/jax_diffusion/trained_models/celeb_a_unet_tiny/unet_tiny_width_64_celeb_a_64_epochs_1000_ema_0.9999_pretraining"
ckpt_name = "checkpoint_last.pkl"
new_ckpt_name = "checkpoint_last_gpu_id_stripped.pkl"

org_path = os.path.join(base_path, ckpt_name)
new_path = os.path.join(base_path, new_ckpt_name)
print(f"Converting {org_path} to {new_path}")

ckpt = dill.load(open(org_path, "rb"))
params = ckpt["params"]

for (k, v) in params.items():
    if isinstance(v, dict):
        for m, z in v.items():
            os = z.shape
            params[k][m] = z[0]
            print(f"Old shape: {z.shape} -> new shape {params[k][m].shape}")
    else:
        params[k] = v[0]
        print(f"Old shape: {v.shape} -> new shape {params[k].shape}")

ckpt["params"] = params
with open(new_path, "wb") as f:
    dill.dump(ckpt, f)


