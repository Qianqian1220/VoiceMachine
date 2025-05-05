import torch
import copy
import os

# —— 修改这两行为你自己的路径 —— 
firered_dir = "/scratch/s6029388/FireRedASR/pretrained_models/FireRedASR-AED-L"
wenet_ckpt = "/scratch/s6029388/wenet/exp/finetune_firered/epoch_20.pt"
output = os.path.join(firered_dir, "model.pth.tar")

# Load original FireRed args if exists
orig = torch.load(output, map_location="cpu", weights_only=False) if os.path.exists(output) else {}
args = orig.get("args")

# Load WenET state_dict
wenet_sd = torch.load(wenet_ckpt, map_location="cpu")

firered_sd = {}
for k, v in wenet_sd.items():
    name = k
    # **Reverse of your mapping rules**
    name = name.replace("encoder.encoders", "encoder.layer_stack")
    name = name.replace(".self_attn.linear_q", ".mhsa.w_qs")
    name = name.replace(".self_attn.linear_k", ".mhsa.w_ks")
    name = name.replace(".self_attn.linear_v", ".mhsa.w_vs")
    name = name.replace(".self_attn.linear_out", ".mhsa.fc")
    name = name.replace(".feed_forward_macaron.w_1", ".ffn1.net.1")
    name = name.replace(".feed_forward_macaron.w_2", ".ffn1.net.4")
    name = name.replace(".feed_forward.w_1", ".ffn2.net.1")
    name = name.replace(".feed_forward.w_2", ".ffn2.net.4")
    # add any additional reverse mappings here as needed

    firered_sd[name] = v

new_ckpt = {"args": args, "model_state_dict": firered_sd}
torch.save(new_ckpt, output)
print("✅ Written FireRedASR checkpoint to", output)