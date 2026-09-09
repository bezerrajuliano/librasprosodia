"""
===========================================================================
 gerar_template_corporal.py  (v4 — corpus 9 temas, Blender 5.2, INT e NEG)
===========================================================================
Le TODAS as actions juliano_NN do .blend (slot OBJECT = Rigify), identifica
tema/tamanho/tipo pela POSICAO da action no corpus (tabela em
config_prosodia.json -> "MAPA_ACTIONS"), forma os pares
    afirmativa (tipo 1)  x  alvo (tipo 2 = INT  |  tipo 3 = NEG)
e, para cada bone corporal candidato, calcula:

    q_delta(t) = q_alvo(t) . q_af(t)^-1      (fase normalizada, N_FASE pts)
    -> vetor de rotacao (log map), eixo dominante, corr_forma intra-tema,
       corr_forma inter-temas, consistencia direcional, pico medio (graus)

Saidas ao lado do .blend (uma dupla por alvo):
    template_corporal_int.csv   estatisticas_corporal_int.csv
    template_corporal_neg.csv   estatisticas_corporal_neg.csv

Schema do template:
    alvo, bone, dom_eixo, corr_intra, corr_inter, direcional, pico_graus,
    fase_idx, fase, rx, ry, rz          (rx,ry,rz em radianos, vetor rotacao)

Config: //config_prosodia.json (criado na 1a execucao; chaves faltantes sao
acrescentadas sem sobrescrever as existentes). NAO altere N_FASE sem
regenerar todos os templates.

Rodar no Editor de Scripts do Blender, com o .blend salvo.
===========================================================================
"""

import bpy
import os
import re
import json
import csv
import math
import numpy as np
from mathutils import Quaternion, Euler

# ==========================================================================
# CONFIG PADRAO
# ==========================================================================
CONFIG_PADRAO = {
    "N_FASE": 101,
    "ORDEM_PAR": "ALVO_MENOS_AF",
    "TIPO_AF": "1",
    "TIPOS_ALVO": {"2": "int"},      # sufixo -> nome de arquivo   # acrescente "3": "neg" na 2a fase
    "TAMANHOS": ["curta", "normal", "longa"],

    # -- Mapeamento das actions (corpus 2026: 9 temas x 3 tam x 3 tipos) ----
    # A action juliano_NN (NN = 1..81) corresponde a:
    #   tema     = ORDEM_TEMAS[(NN-1) // 9]
    #   tamanho  = TAMANHOS[((NN-1) % 9) // 3]
    #   tipo     = ((NN-1) % 3) + 1      (1 af, 2 int, 3 neg)
    # Se a ordem real for outra, edite MAPA_ACTIONS no JSON (tem prioridade).
    "PADRAO_ACTION": "juliano_{:02d}",
    "ORDEM_TEMAS": ["viagem", "comida", "estudo", "compra", "passeio",
                    "vestido", "livro", "cafe", "futebol"],
    "MAPA_ACTIONS": {},                          # preenchido na 1a execucao

    # -- Limiares unificados (corpo E face) ------------------------------------
    "CORR_MINIMA": 0.50,
    "DIRECIONAL_MIN": 0.80,

    # -- Corpo -----------------------------------------------------------------
    "ARMATURE": "",                              # vazio = detectar
    "CORPO_TERMOS_BONES": ["neck", "head", "spine_fk", "chest", "torso",
                           "tweak_spine", "shoulder", "hips"],
    "CORPO_PREFIXOS_EXCLUIR": ["CTRL_", "DEF-", "MCH-", "ORG-", "VIS_"],
    "CORPO_MIN_ANGULO_GRAUS": 2.0,

    # -- Facial (usado pelo outro script) --------------------------------------
    "FACIAL_PREFIXO_CTRL": "CTRL_",
    "FACIAL_MIN_AMPLITUDE": 0.010,
    "FACIAL_INCLUIR_ARTICULACAO": False,
    "FACIAL_TERMOS_ARTICULACAO": ["jaw", "mouth", "lip", "corner", "tongue",
                                  "chin", "teeth", "neck_", "ear"],
    "FACIAL_SIMETRIA_FORCADA": True,
}

ARQ_CONFIG = "config_prosodia.json"


# ==========================================================================
# CONFIG / MAPA
# ==========================================================================
def base_dir():
    b = bpy.path.abspath("//")
    if not b:
        raise RuntimeError("Salve o .blend antes de rodar.")
    return b


def gerar_mapa(cfg):
    mapa = {}
    tams = cfg["TAMANHOS"]
    temas = cfg["ORDEM_TEMAS"]
    for n in range(1, len(temas) * 9 + 1):
        i = n - 1
        mapa[cfg["PADRAO_ACTION"].format(n)] = "{}_{}_{}".format(
            temas[i // 9], tams[(i % 9) // 3], (i % 3) + 1)
    return mapa


def carregar_config():
    p = os.path.join(base_dir(), ARQ_CONFIG)
    cfg = dict(CONFIG_PADRAO)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    if not cfg.get("MAPA_ACTIONS"):
        cfg["MAPA_ACTIONS"] = gerar_mapa(cfg)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("Config:", p)
    return cfg


RE_UNIDADE = re.compile(r"^(?P<tema>[a-z0-9]+)_(?P<tam>curta|normal|longa)_(?P<tipo>[123])$")


def parse_unidade(rotulo):
    m = RE_UNIDADE.match(rotulo.lower())
    return (m.group("tema"), m.group("tam"), m.group("tipo")) if m else None


# ==========================================================================
# ACESSO AS FCURVES (slotted actions, sem atribuir nada ao rig)
# ==========================================================================
def channelbag_object(act):
    """Retorna (slot, channelbag) do primeiro slot OBJECT da action."""
    for slot in act.slots:
        if slot.target_id_type != "OBJECT":
            continue
        for layer in act.layers:
            for strip in layer.strips:
                cb = strip.channelbag(slot)
                if cb is not None:
                    return slot, cb
    return None, None


def frame_range(act, cb):
    fr = act.frame_range
    if fr[1] > fr[0]:
        return float(fr[0]), float(fr[1])
    lo, hi = math.inf, -math.inf
    for fc in cb.fcurves:
        r = fc.range()
        lo, hi = min(lo, r[0]), max(hi, r[1])
    return lo, hi


RE_BONE = re.compile(r'pose\.bones\["(?P<bone>[^"]+)"\]\.(?P<prop>\w+)$')


def indexar_fcurves(cb):
    """{bone: {prop: {index: fcurve}}}"""
    idx = {}
    for fc in cb.fcurves:
        m = RE_BONE.match(fc.data_path)
        if not m:
            continue
        idx.setdefault(m.group("bone"), {}).setdefault(m.group("prop"), {})[fc.array_index] = fc
    return idx


def amostrar(fcs, n_comp, padrao, frames):
    """Avalia n_comp componentes em frames (float). padrao = valor se canal ausente."""
    out = np.zeros((len(frames), n_comp))
    for i in range(n_comp):
        fc = fcs.get(i)
        if fc is None:
            out[:, i] = padrao[i]
        else:
            out[:, i] = [fc.evaluate(f) for f in frames]
    return out


def quats_bone(bone_fcs, modo_rot, frames):
    """Retorna array (n,4) de quaternions unitarios na fase."""
    if "rotation_quaternion" in bone_fcs and modo_rot == "QUATERNION":
        q = amostrar(bone_fcs["rotation_quaternion"], 4, (1, 0, 0, 0), frames)
    elif "rotation_euler" in bone_fcs:
        e = amostrar(bone_fcs["rotation_euler"], 3, (0, 0, 0), frames)
        ordem = modo_rot if modo_rot in {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"} else "XYZ"
        q = np.array([Euler(tuple(r), ordem).to_quaternion()[:] for r in e])
    elif "rotation_quaternion" in bone_fcs:
        q = amostrar(bone_fcs["rotation_quaternion"], 4, (1, 0, 0, 0), frames)
    else:
        return None
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n == 0] = 1
    q = q / n
    # continuidade de sinal (evita saltos q <-> -q)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def delta_rotvec(q_alvo, q_af):
    """q_delta = q_alvo . q_af^-1 -> vetor de rotacao (eixo*angulo), (n,3)."""
    out = np.zeros((len(q_af), 3))
    for i in range(len(q_af)):
        qa = Quaternion(q_alvo[i]); qb = Quaternion(q_af[i])
        qd = qa @ qb.inverted()
        if qd.w < 0:
            qd = -qd
        ax, ang = qd.to_axis_angle()
        out[i] = np.array(ax) * ang
    return out


# ==========================================================================
# METRICAS
# ==========================================================================
def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def media_pares(curvas):
    if len(curvas) < 2:
        return float("nan")
    vals = [pearson(curvas[i], curvas[j])
            for i in range(len(curvas)) for j in range(i + 1, len(curvas))]
    return float(np.mean(vals))


def avaliar_marcador(deltas, cfg):
    """
    deltas: {(tema, tam): array (N,3)}  -> dict de metricas + curva media.
    """
    todas = np.stack(list(deltas.values()))              # (P,N,3)
    media = todas.mean(axis=0)                           # (N,3)
    dom = int(np.argmax(np.abs(media).max(axis=0)))      # eixo dominante
    pico_idx = int(np.argmax(np.abs(media[:, dom])))
    sinal = np.sign(media[pico_idx, dom]) or 1.0

    # intra-tema: correlacao entre tamanhos do mesmo tema
    por_tema = {}
    for (tema, tam), d in deltas.items():
        por_tema.setdefault(tema, []).append(d[:, dom])
    intra = [media_pares(c) for c in por_tema.values() if len(c) >= 2]
    corr_intra = float(np.mean(intra)) if intra else float("nan")

    # inter-temas: correlacao entre curvas medias de cada tema
    medias_tema = [np.mean(c, axis=0) for c in por_tema.values()]
    corr_inter = media_pares(medias_tema)

    # direcional: fracao de pares cujo valor no pico medio tem o mesmo sinal
    concord = [np.sign(d[pico_idx, dom]) == sinal for d in deltas.values()]
    direcional = float(np.mean(concord))

    pico_graus = float(np.degrees(np.abs(media[pico_idx, dom])))
    pico_medio_pares = float(np.degrees(np.mean(
        [np.abs(d[:, dom]).max() for d in deltas.values()])))

    ok = (corr_intra >= cfg["CORR_MINIMA"] and corr_inter >= cfg["CORR_MINIMA"]
          and direcional >= cfg["DIRECIONAL_MIN"]
          and pico_graus >= cfg["CORPO_MIN_ANGULO_GRAUS"])
    return dict(media=media, dom_eixo="XYZ"[dom], corr_intra=corr_intra,
                corr_inter=corr_inter, direcional=direcional,
                pico_graus=pico_graus, pico_medio_pares=pico_medio_pares,
                fase_pico=pico_idx / (len(media) - 1), n_pares=len(deltas),
                aprovado=ok)


# ==========================================================================
# MAIN
# ==========================================================================
def bone_candidato(nome, cfg):
    if any(nome.startswith(p) for p in cfg["CORPO_PREFIXOS_EXCLUIR"]):
        return False
    return any(t in nome for t in cfg["CORPO_TERMOS_BONES"])


def achar_armature(cfg):
    if cfg.get("ARMATURE"):
        return bpy.data.objects[cfg["ARMATURE"]]
    for o in bpy.data.objects:
        if o.type == "ARMATURE" and "head" in o.pose.bones and "spine_fk" in o.pose.bones:
            return o
    raise RuntimeError("Armature Rigify nao encontrado; defina ARMATURE no config.")


def main():
    cfg = carregar_config()
    N = cfg["N_FASE"]
    fases = np.linspace(0.0, 1.0, N)
    arm = achar_armature(cfg)
    modos = {pb.name: pb.rotation_mode for pb in arm.pose.bones}
    print("Armature:", arm.name)

    # 1) coleta: unidade -> {bone: quats(N,4)}
    unidades = {}
    for act in bpy.data.actions:
        rot = cfg["MAPA_ACTIONS"].get(act.name)
        if not rot:
            continue
        u = parse_unidade(rot)
        if not u:
            print("  rotulo invalido no MAPA_ACTIONS:", act.name, "->", rot)
            continue
        slot, cb = channelbag_object(act)
        if cb is None:
            print("  sem slot OBJECT:", act.name)
            continue
        f0, f1 = frame_range(act, cb)
        frames = f0 + fases * (f1 - f0)
        idx = indexar_fcurves(cb)
        dados = {}
        for bone, fcs in idx.items():
            if not bone_candidato(bone, cfg):
                continue
            q = quats_bone(fcs, modos.get(bone, "QUATERNION"), frames)
            if q is not None:
                dados[bone] = q
        unidades[u] = dados
        print(f"  {act.name:12s} -> {rot:18s} frames {f0:.0f}-{f1:.0f}  bones {len(dados)}")

    if not unidades:
        raise RuntimeError("Nenhuma action mapeada. Confira MAPA_ACTIONS no config.")

    # 2) para cada alvo (int / neg)
    for tipo_alvo, nome_alvo in cfg["TIPOS_ALVO"].items():
        pares = []
        for (tema, tam, tipo) in unidades:
            if tipo == cfg["TIPO_AF"] and (tema, tam, tipo_alvo) in unidades:
                pares.append((tema, tam))
        print(f"\n=== ALVO {nome_alvo.upper()} (tipo {tipo_alvo}): {len(pares)} pares ===")
        if not pares:
            continue

        bones = set()
        for tema, tam in pares:
            bones |= set(unidades[(tema, tam, cfg["TIPO_AF"])]) & set(unidades[(tema, tam, tipo_alvo)])

        stats, template = [], []
        for bone in sorted(bones):
            deltas = {}
            for tema, tam in pares:
                qa = unidades[(tema, tam, cfg["TIPO_AF"])].get(bone)
                qb = unidades[(tema, tam, tipo_alvo)].get(bone)
                if qa is None or qb is None:
                    continue
                deltas[(tema, tam)] = delta_rotvec(qb, qa)
            if len(deltas) < 2:
                continue
            m = avaliar_marcador(deltas, cfg)
            stats.append(dict(alvo=nome_alvo, bone=bone, n_pares=m["n_pares"],
                              dom_eixo=m["dom_eixo"],
                              corr_intra=round(m["corr_intra"], 4),
                              corr_inter=round(m["corr_inter"], 4),
                              direcional=round(m["direcional"], 3),
                              pico_graus=round(m["pico_graus"], 3),
                              pico_medio_pares_graus=round(m["pico_medio_pares"], 3),
                              fase_pico=round(m["fase_pico"], 3),
                              aprovado=int(m["aprovado"])))
            if m["aprovado"]:
                for k in range(N):
                    template.append(dict(alvo=nome_alvo, bone=bone, dom_eixo=m["dom_eixo"],
                                         corr_intra=round(m["corr_intra"], 4),
                                         corr_inter=round(m["corr_inter"], 4),
                                         direcional=round(m["direcional"], 3),
                                         pico_graus=round(m["pico_graus"], 3),
                                         fase_idx=k, fase=round(fases[k], 4),
                                         rx=round(float(m["media"][k, 0]), 6),
                                         ry=round(float(m["media"][k, 1]), 6),
                                         rz=round(float(m["media"][k, 2]), 6)))

        stats.sort(key=lambda r: (-r["aprovado"], -r["pico_graus"]))
        b = base_dir()
        for nome, linhas in ((f"estatisticas_corporal_{nome_alvo}.csv", stats),
                             (f"template_corporal_{nome_alvo}.csv", template)):
            if not linhas:
                print("  (vazio)", nome); continue
            with open(os.path.join(b, nome), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
                w.writeheader(); w.writerows(linhas)
            print("  gravado:", nome, f"({len(linhas)} linhas)")

        print(f"\n  {'bone':18s} eixo  intra   inter   dir    pico(g)  fase  ok")
        for r in stats:
            print(f"  {r['bone']:18s}  {r['dom_eixo']}   {r['corr_intra']:6.3f}  "
                  f"{r['corr_inter']:6.3f}  {r['direcional']:4.2f}  {r['pico_graus']:7.2f}  "
                  f"{r['fase_pico']:4.2f}  {'*' if r['aprovado'] else ''}")
        print(f"  aprovados: {sum(r['aprovado'] for r in stats)} de {len(stats)}")


main()
