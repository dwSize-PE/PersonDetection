# app/osnet/osnet_model.py
from __future__ import annotations

import os
import warnings
from typing import List, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from app.osnet_external.osnet import osnet_x1_0


class OsNetEmbedder:
    def __init__(self, weight_path: str = "models/osnet_ibn_x1_0_imagenet.pth", device: str = "auto", half: bool = False):
        self.weight_path = weight_path
        self.device = self._select_device(device)
        self.half = bool(half and self.device.type == "cuda")

        num_classes = 1000  # padrão
        if os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
            new_state = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
            if 'classifier.weight' in new_state:
                num_classes = new_state['classifier.weight'].shape[0]

        self.model = osnet_x1_0(num_classes=num_classes, pretrained=False, loss="triplet")
        self.model.eval().to(self.device)
        if self.half:
            self.model.half()

        self._load_weights(self.weight_path)

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        try:
            self._warmup()
        except Exception:
            pass

    @torch.inference_mode()
    def extract_one(self, bgr_image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extrai embedding de uma imagem BGR.
        
        Retorna Tensor (512,), CPU, float32, L2 normalizado.
        
        Parâmetros
        ----------
        bgr_image : np.ndarray
            Imagem BGR (H, W, 3)
        
        Retorna
        -------
        embedding : torch.Tensor | None
            Tensor (512,) L2-normalizado ou None se falhar
        """
        tensor = self._preprocess_bgr(bgr_image)
        if tensor is None:
            return None
        feat = self._forward_tensor(tensor)  # torch.Tensor (512,)
        return feat

    @torch.inference_mode()
    def extract(self, bgr_crops: List[np.ndarray]) -> np.ndarray:
        """
        Mantido: retorna np.ndarray (N,512) para usos batch atuais (opcional).
        
        Parâmetros
        ----------
        bgr_crops : list[np.ndarray]
            Lista de imagens BGR
        
        Retorna
        -------
        embeddings : np.ndarray
            Array (N, 512) de embeddings normalizados
        """
        if not bgr_crops:
            return np.zeros((0, 512), dtype=np.float32)

        tensors = []
        valids = []
        for img in bgr_crops:
            t = self._preprocess_bgr(img)
            if t is None:
                valids.append(False)
            else:
                valids.append(True)
                tensors.append(t)

        feats = []
        if tensors:
            batch = torch.stack(tensors, dim=0)  # (M,3,256,128)
            if self.half:
                batch = batch.half()
            batch = batch.to(self.device, non_blocking=True)
            v = self.model(batch)                      # (M,512)
            
            # ============================================================
            # NORMALIZA L2 (única normalização necessária)
            # ============================================================
            v = torch.nn.functional.normalize(v, p=2, dim=1)
            feats_batch = v.detach().float().cpu().numpy()
        else:
            feats_batch = np.zeros((0, 512), dtype=np.float32)

        it = iter(feats_batch)
        for ok in valids:
            if ok:
                feats.append(next(it))
            else:
                feats.append(np.zeros((512,), dtype=np.float32))
        return np.stack(feats, axis=0)

    def _select_device(self, device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _load_weights(self, path: str) -> None:
        if not os.path.exists(path):
            warnings.warn(f"[OsNetEmbedder] Pesos não encontrados em '{path}'. O modelo rodará SEM pesos pré-treinados.")
            return
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get('state_dict', ckpt)
        new_state = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith('module.') else k
            new_state[nk] = v
        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        if missing:
            warnings.warn(f"[OsNetEmbedder] Camadas ausentes: {sorted(missing)[:5]} ...")
        if unexpected:
            warnings.warn(f"[OsNetEmbedder] Pesos inesperados: {sorted(unexpected)[:5]} ...")

    def _preprocess_bgr(self, bgr: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            return None
        h, w = bgr.shape[:2]
        if h < 16 or w < 16:
            return None
        rgb = bgr[:, :, ::-1]
        img = Image.fromarray(rgb)
        t = cast(torch.Tensor, self.transform(img))   # (3,256,128), float32
        return t

    def _forward_tensor(self, tensor_3chw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do OSNet.
        
        Retorna embedding (512,) L2-normalizado.
        
        Parâmetros
        ----------
        tensor_3chw : torch.Tensor
            Tensor (3, 256, 128) preprocessado
        
        Retorna
        -------
        embedding : torch.Tensor
            Tensor (512,) L2-normalizado, CPU, float32
        """
        if self.half:
            tensor_3chw = tensor_3chw.half()
        x = tensor_3chw.unsqueeze(0).to(self.device, non_blocking=True)  # (1,3,256,128)
        v = self.model(x)                                                # (1,512)
        
        # ============================================================
        # NORMALIZA L2 (única normalização necessária)
        # Modelo OSNet retorna embedding cru, normalização é feita aqui
        # ============================================================
        v = torch.nn.functional.normalize(v, p=2, dim=1)
        
        # ============================================================
        # LOG DEBUG (opcional - comentar em produção)
        # ============================================================
        # print(f"[OSNET_NORM] norm={torch.norm(v, p=2, dim=1).item():.4f} mean={v.mean().item():.4f}")
        
        return v.detach().float().cpu()[0]  # torch.Tensor (512,)

    def _warmup(self):
        """Warmup do modelo (opcional)"""
        dummy = torch.zeros(1, 3, 256, 128)
        if self.half:
            dummy = dummy.half()
        dummy = dummy.to(self.device)
        for _ in range(3):
            _ = self.model(dummy)