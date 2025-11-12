import torch
import os
import torch.nn as nn
from app.osnet_external.osnet import osnet_ibn_x1_0  # ajuste o import se necessário

src_path = "models/best_model.pth"
dst_path = "models/best_model_fixed_final.pth"

print(f"[INFO] Carregando checkpoint: {src_path}")
ckpt = torch.load(src_path, map_location="cpu")
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# Instancia o modelo base (sem pesos) para identificar camadas InstanceNorm2d
model = osnet_ibn_x1_0(num_classes=1453, pretrained=False, loss="softmax")
instance_layers = []

for name, module in model.named_modules():
    if isinstance(module, nn.InstanceNorm2d):
        instance_layers.append(name)

print(f"[INFO] Detectadas {len(instance_layers)} camadas InstanceNorm2d")

# Remove apenas as stats dessas camadas
new_state = {}
removed = []
for key, val in state_dict.items():
    match = any(key.startswith(layer) and (".running_mean" in key or ".running_var" in key or ".num_batches_tracked" in key)
                for layer in instance_layers)
    if match:
        removed.append(key)
        continue
    new_state[key] = val

if "state_dict" in ckpt:
    ckpt["state_dict"] = new_state
else:
    ckpt = new_state

os.makedirs(os.path.dirname(dst_path), exist_ok=True)
torch.save(ckpt, dst_path)

print("\n✅ Checkpoint corrigido de forma inteligente!")
print(f"   → Salvo em: {dst_path}")
print(f"   → Stats removidos: {len(removed)}")
if removed:
    print("   → Exemplo:", removed[:5])
