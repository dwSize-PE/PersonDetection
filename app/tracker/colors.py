# ================================================
# colors.py — cores persistentes por ID global
# ================================================
import random

_color_map = {}

def get_color(id_num: int):
    """
    Retorna cor BGR persistente para um ID global.
    Cores são geradas aleatoriamente na primeira vez e mantidas.
    """
    if id_num not in _color_map:
        _color_map[id_num] = (
            random.randint(60, 255),
            random.randint(60, 255),
            random.randint(60, 255),
        )
    return _color_map[id_num]


def reset_colors():
    """
    Limpa todas as cores (útil para reset completo do sistema)
    """
    global _color_map
    _color_map = {}