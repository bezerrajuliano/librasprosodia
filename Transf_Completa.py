"""
===========================================================================
 transformacao_completa_slot.py  (v4.3 — fake user nas copias)
===========================================================================
Sintetiza a versao prosodica (INT por padrao) de cada frase AFIRMATIVA
aplicando os templates gerados:

    corpo : q_sint(f) = q_delta(fase)^k . q_af(f)     (k = intensidade)
    face  : loc_sint(f) = loc_af(f) + k * delta(fase)  nos CTRL_* aprovados

Entradas (na pasta do .blend): config_prosodia.json,
template_corporal_<alvo>.csv, template_facial_<alvo>.csv.

Para cada action afirmativa (tipo TIPO_AF em MAPA_ACTIONS) cria uma COPIA
"<action>_<TRANSF_SUFIXO_SLOT>" (ex.: viagem_curta_afirmativa_sint) e faz o
bake das curvas modificadas nela, a cada PASSO_FRAMES. As actions originais
nao sao tocadas. Rodar de novo substitui as copias.

--- v4.3 -----------------------------------------------------------------
5. use_fake_user=True nas copias sinteticas: sem isso elas ficam com zero
   usuarios e o Blender nao as grava ao salvar — sumiam ao reabrir.

--- MUDANCAS DA v4.1 PARA A v4.2 -----------------------------------------
1. resolver_action(): o nome real da action no .blend passa a ser procurado
   em varias formas — a chave do MAPA_ACTIONS, o rotulo, e a conversao do
   sufixo numerico para extenso (viagem_curta_1 -> viagem_curta_afirmativa).
   Assim o script funciona com as actions ainda como "juliano_NN" OU ja
   renomeadas, sem depender de qual formato esta no config.
2. canon_tipo(): "1"/"af"/"afirmativa" sao equivalentes, idem 2/int e
   3/neg. TIPO_AF e TRANSF_ALVO podem estar em qualquer um dos formatos.
3. limites_corpus() usa o mesmo resolvedor — sem isso o clamp "corpus"
   varreria zero actions e silenciosamente viraria clamp nenhum.
4. Log inicial dos valores efetivos do config (lembrando que o JSON sempre
   tem precedencia sobre PADRAO_TRANSF) e relatorio final de ausentes.

Chaves de controle (config_prosodia.json):
  TRANSF_ALVO           "int" / "interrogativa" (ou "neg" na 2a fase)
  TRANSF_ALVOS          "todas" ou lista de unidades ["viagem_curta", ...]
  TRANSF_INTENSIDADE_CABECA / _TRONCO   escala do delta corporal
  TRANSF_TERMOS_CABECA  bones tratados como cabeca (demais = tronco)
  TRANSF_INTENSIDADE_FACIAL             escala do delta facial
  FACIAL_SOMENTE_EIXO_DOMINANTE         aplica so o eixo dominante do CTRL
  FACIAL_CLAMP "corpus" (recomendado): limita cada canal ao intervalo que o
      sinalizante percorreu em TODO o corpus capturado (+- margem) — permite
      adicionar prosodia a canais parados na afirmativa sem jamais extrapolar
      a face real; "observado": intervalo apenas da propria afirmativa
      (ATENCAO: zera o delta de canais parados na afirmativa); "fixo": usa
      FACIAL_CLAMP_MIN/MAX; "nenhum": sem limite
  TRANSF_ATRIBUIR_AO_RIG  atribui a 1a copia ao rig para conferencia

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

ARQ_CONFIG = "config_prosodia.json"

PADRAO_TRANSF = {
    "TRANSF_ALVO": "int",
    "TRANSF_ALVOS": "todas",
    "TRANSF_SUFIXO_SLOT": "sint",
    "TRANSF_INTENSIDADE_CABECA": 1.3,
    "TRANSF_INTENSIDADE_TRONCO": 1.0,
    "TRANSF_TERMOS_CABECA": ["head", "neck"],
    "TRANSF_INTENSIDADE_FACIAL": 1.0,
    "FACIAL_SOMENTE_EIXO_DOMINANTE": True,
    "FACIAL_CLAMP": "corpus",
    "FACIAL_CLAMP_MARGEM": 0.15,
    "FACIAL_CLAMP_MIN": -1.0,
    "FACIAL_CLAMP_MAX": 1.0,
    "PASSO_FRAMES": 1,
    "TRANSF_ATRIBUIR_AO_RIG": True,
}

# --------------------------------------------------------------------------
# VOCABULARIO DE TIPOS  (v4.2)
# --------------------------------------------------------------------------
# O corpus usava sufixo numerico (_1/_2/_3); as actions do .blend passaram a
# usar o nome por extenso. Tudo e reduzido a uma forma canonica para que os
# dois formatos convivam sem quebrar os geradores de template.
CANON_TIPO = {
    "1": "afirmativa", "af": "afirmativa", "afir": "afirmativa",
    "afirmativa": "afirmativa",
    "2": "interrogativa", "int": "interrogativa",
    "interrogativa": "interrogativa",
    "3": "negativa", "neg": "negativa", "negativa": "negativa",
}

# sufixo numerico -> nome por extenso, para reconstruir o nome da action
NUM_PARA_EXTENSO = {"1": "afirmativa", "2": "interrogativa", "3": "negativa"}


def canon_tipo(t):
    """'1', 'af', 'afirmativa' -> 'afirmativa' (idem 2/int e 3/neg)."""
    s = str(t).strip().lower()
    return CANON_TIPO.get(s, s)


def variantes_nome(nome):
    """Formas alternativas do mesmo nome, trocando o sufixo de tipo."""
    saida = [nome]
    base, _, suf = nome.rpartition("_")
    if base:
        s = suf.lower()
        if s in NUM_PARA_EXTENSO:
            saida.append(base + "_" + NUM_PARA_EXTENSO[s])
        else:
            for num, ext in NUM_PARA_EXTENSO.items():
                if s == ext:
                    saida.append(base + "_" + num)
    return saida


def nome_posicional(cfg, rotulo):
    """Reconstroi 'juliano_NN' pela posicao do item no corpus.

    indice = tema*9 + tamanho*3 + tipo, na ordem declarada em ORDEM_TEMAS,
    TAMANHOS e (afirmativa, interrogativa, negativa). Ultimo recurso, para
    abrir um .blend antigo (ainda sem renomear) com o config novo.
    """
    padrao = cfg.get("PADRAO_ACTION")
    temas = cfg.get("ORDEM_TEMAS")
    tams = cfg.get("TAMANHOS")
    if not padrao or not temas or not tams:
        return None
    partes = rotulo.split("_")
    if len(partes) < 3:
        return None
    tema, tam, tipo = partes[0], partes[1], canon_tipo(partes[2])
    ordem_tipos = ["afirmativa", "interrogativa", "negativa"]
    if tema not in temas or tam not in tams or tipo not in ordem_tipos:
        return None
    idx = (temas.index(tema) * len(tams) * 3
           + tams.index(tam) * 3
           + ordem_tipos.index(tipo) + 1)
    try:
        return padrao.format(idx)
    except Exception:
        return None


def resolver_action(chave, rotulo, cfg=None):
    """Acha a action real no .blend a partir da chave e/ou do rotulo do mapa.

    Retorna (nome_real, action) ou (None, None). Aceita o .blend com as
    actions ainda como 'juliano_NN' ou ja renomeadas, em qualquer combinacao
    com o formato usado no config.
    """
    candidatos = []
    for n in (chave, rotulo):
        for v in variantes_nome(n):
            if v not in candidatos:
                candidatos.append(v)
    if cfg is not None:
        pos = nome_posicional(cfg, rotulo)
        if pos and pos not in candidatos:
            candidatos.append(pos)
    for nome in candidatos:
        act = bpy.data.actions.get(nome)
        if act is not None:
            return nome, act
    return None, None


def base_dir():
    b = bpy.path.abspath("//")
    if not b:
        raise RuntimeError("Salve o .blend antes de rodar.")
    return b


def carregar_config():
    p = os.path.join(base_dir(), ARQ_CONFIG)
    if not os.path.exists(p):
        raise RuntimeError("config_prosodia.json nao encontrado — rode os geradores antes.")
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    novo = False
    for k, v in PADRAO_TRANSF.items():
        if k not in cfg:
            cfg[k] = v; novo = True
    if novo:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


# ==========================================================================
# TEMPLATES
# ==========================================================================
def ler_template(caminho, comps):
    """{bone: {"fase": (N,), "val": (N,3), "dom": idx_eixo}}"""
    if not os.path.exists(caminho):
        raise RuntimeError("template nao encontrado: " + caminho)
    bruto = {}
    with open(caminho, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = bruto.setdefault(r["bone"], {"fase": [], "val": [], "dom": r["dom_eixo"]})
            b["fase"].append(float(r["fase"]))
            b["val"].append([float(r[c]) for c in comps])
    out = {}
    for bone, d in bruto.items():
        ordem = np.argsort(d["fase"])
        out[bone] = {"fase": np.array(d["fase"])[ordem],
                     "val": np.array(d["val"])[ordem],
                     "dom": "XYZ".index(d["dom"])}
    return out


def delta_na_fase(tpl, fases):
    """Interpola linearmente o template nas fases pedidas -> (n,3)."""
    out = np.empty((len(fases), 3))
    for c in range(3):
        out[:, c] = np.interp(fases, tpl["fase"], tpl["val"][:, c])
    return out


# ==========================================================================
# FCURVES / SLOTS
# ==========================================================================
def channelbag_object(act):
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
        r = fc.range(); lo, hi = min(lo, r[0]), max(hi, r[1])
    return lo, hi


RE_BONE = re.compile(r'pose\.bones\["(?P<bone>[^"]+)"\]\.(?P<prop>\w+)$')


def indexar(cb):
    idx = {}
    for fc in cb.fcurves:
        m = RE_BONE.match(fc.data_path)
        if m:
            idx.setdefault(m.group("bone"), {}).setdefault(m.group("prop"), {})[fc.array_index] = fc
    return idx


def amostrar(fcs, n_comp, padrao, frames):
    out = np.zeros((len(frames), n_comp))
    for i in range(n_comp):
        fc = fcs.get(i)
        out[:, i] = padrao[i] if fc is None else [fc.evaluate(f) for f in frames]
    return out


def gravar(cb, data_path, index, frames, valores):
    """Substitui/cria a fcurve e grava os pares (frame, valor) com bake linear."""
    fc = cb.fcurves.find(data_path, index=index)
    if fc is not None:
        cb.fcurves.remove(fc)
    fc = cb.fcurves.new(data_path, index=index)
    fc.keyframe_points.add(len(frames))
    co = np.empty(len(frames) * 2)
    co[0::2] = frames; co[1::2] = valores
    fc.keyframe_points.foreach_set("co", co)
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"
    fc.update()


# ==========================================================================
# TRANSFORMACAO
# ==========================================================================
def intensidade_corpo(bone, cfg):
    if any(t in bone for t in cfg["TRANSF_TERMOS_CABECA"]):
        return float(cfg["TRANSF_INTENSIDADE_CABECA"])
    return float(cfg["TRANSF_INTENSIDADE_TRONCO"])


def q_do_rotvec(v, escala):
    ang = float(np.linalg.norm(v)) * escala
    if ang < 1e-9:
        return Quaternion((1, 0, 0, 0))
    eixo = v / np.linalg.norm(v)
    return Quaternion(tuple(eixo), ang)


def aplicar_corpo(idx, cb, bone, fcs, tpl, frames, fases, modo_rot, cfg):
    k = intensidade_corpo(bone, cfg)
    if k == 0.0:
        return False
    delta = delta_na_fase(tpl, fases)
    if "rotation_quaternion" in fcs and modo_rot == "QUATERNION":
        q0 = amostrar(fcs["rotation_quaternion"], 4, (1, 0, 0, 0), frames)
        for i in range(1, len(q0)):
            if np.dot(q0[i], q0[i - 1]) < 0:
                q0[i] = -q0[i]
        novo = np.empty_like(q0)
        for i in range(len(frames)):
            q = q_do_rotvec(delta[i], k) @ Quaternion(q0[i])
            novo[i] = q[:]
        dp = f'pose.bones["{bone}"].rotation_quaternion'
        for c in range(4):
            gravar(cb, dp, c, frames, novo[:, c])
    elif "rotation_euler" in fcs:
        ordem = modo_rot if modo_rot in {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"} else "XYZ"
        e0 = amostrar(fcs["rotation_euler"], 3, (0, 0, 0), frames)
        novo = np.empty_like(e0)
        for i in range(len(frames)):
            q = q_do_rotvec(delta[i], k) @ Euler(tuple(e0[i]), ordem).to_quaternion()
            novo[i] = q.to_euler(ordem)[:]
        dp = f'pose.bones["{bone}"].rotation_euler'
        for c in range(3):
            gravar(cb, dp, c, frames, novo[:, c])
    else:
        return False
    return True


def limites_corpus(cfg):
    """{bone: (min(3), max(3))} varrendo os keyframes de todas as actions mapeadas."""
    lim = {}
    achadas = 0
    for chave, rotulo in cfg["MAPA_ACTIONS"].items():
        _, act = resolver_action(chave, rotulo, cfg)   # v4.2: resolve o nome real
        if act is None:
            continue
        achadas += 1
        _, cb = channelbag_object(act)
        if cb is None:
            continue
        for fc in cb.fcurves:
            m = RE_BONE.match(fc.data_path)
            if not m or m.group("prop") != "location" or not m.group("bone").startswith("CTRL_"):
                continue
            n = len(fc.keyframe_points)
            if n == 0:
                continue
            co = np.empty(n * 2)
            fc.keyframe_points.foreach_get("co", co)
            ys = co[1::2]
            b, c = m.group("bone"), fc.array_index
            lo, hi = lim.setdefault(b, [np.full(3, np.inf), np.full(3, -np.inf)])
            lo[c] = min(lo[c], ys.min()); hi[c] = max(hi[c], ys.max())
    if achadas == 0:
        raise RuntimeError(
            "clamp 'corpus': nenhuma action do MAPA_ACTIONS foi encontrada no .blend. "
            "Confira os nomes — sem isso o clamp nao teria efeito nenhum."
        )
    print(f"clamp 'corpus': {achadas}/{len(cfg['MAPA_ACTIONS'])} actions varridas")
    return lim


def aplicar_face(cb, bone, fcs, tpl, frames, fases, cfg, lim_corpus=None):
    k = float(cfg["TRANSF_INTENSIDADE_FACIAL"])
    if k == 0.0 or "location" not in fcs:
        return False
    delta = delta_na_fase(tpl, fases) * k
    if cfg.get("FACIAL_SOMENTE_EIXO_DOMINANTE", True):
        m = np.zeros(3); m[tpl["dom"]] = 1.0
        delta = delta * m
    loc0 = amostrar(fcs["location"], 3, (0, 0, 0), frames)
    novo = loc0 + delta
    modo = cfg.get("FACIAL_CLAMP", "corpus")
    for c in range(3):
        if modo == "corpus" and lim_corpus and bone in lim_corpus:
            lo, hi = lim_corpus[bone][0][c], lim_corpus[bone][1][c]
            marg = cfg["FACIAL_CLAMP_MARGEM"] * max(hi - lo, 1e-4)
            novo[:, c] = np.clip(novo[:, c], lo - marg, hi + marg)
        elif modo == "observado":
            lo, hi = loc0[:, c].min(), loc0[:, c].max()
            marg = cfg["FACIAL_CLAMP_MARGEM"] * max(hi - lo, 1e-4)
            novo[:, c] = np.clip(novo[:, c], lo - marg, hi + marg)
        elif modo == "fixo":
            novo[:, c] = np.clip(novo[:, c], cfg["FACIAL_CLAMP_MIN"], cfg["FACIAL_CLAMP_MAX"])
    dp = f'pose.bones["{bone}"].location'
    for c in range(3):
        gravar(cb, dp, c, frames, novo[:, c])
    return True


def main():
    cfg = carregar_config()
    alvo = cfg.get("TRANSF_ALVO", "int")
    tipo_af = canon_tipo(cfg.get("TIPO_AF", "1"))
    b = base_dir()

    # v4.2: o JSON sempre tem precedencia — imprimir os valores efetivos
    print("\n" + "=" * 70)
    print("valores efetivos do config_prosodia.json")
    print("  TIPO_AF ............... {!r}  (canonico: {})".format(cfg.get("TIPO_AF", "1"), tipo_af))
    print("  TRANSF_ALVO ........... {!r}".format(alvo))
    print("  TRANSF_ALVOS .......... {!r}".format(cfg.get("TRANSF_ALVOS", "todas")))
    print("  INTENSIDADE cabeca .... {}".format(cfg["TRANSF_INTENSIDADE_CABECA"]))
    print("  INTENSIDADE tronco .... {}".format(cfg["TRANSF_INTENSIDADE_TRONCO"]))
    print("  INTENSIDADE facial .... {}".format(cfg["TRANSF_INTENSIDADE_FACIAL"]))
    print("  FACIAL_CLAMP .......... {!r}".format(cfg.get("FACIAL_CLAMP", "corpus")))
    print("=" * 70)

    # nome de arquivo do template: aceita alvo por extenso ou abreviado
    curto = {"afirmativa": "af", "interrogativa": "int", "negativa": "neg"}
    alvo_arq = curto.get(canon_tipo(alvo), str(alvo))
    tpl_c = ler_template(os.path.join(b, f"template_corporal_{alvo_arq}.csv"), ["rx", "ry", "rz"])
    tpl_f = ler_template(os.path.join(b, f"template_facial_{alvo_arq}.csv"), ["dx", "dy", "dz"])
    print(f"templates ({alvo_arq}): {len(tpl_c)} juntas, {len(tpl_f)} controles faciais")

    arm = None
    for o in bpy.data.objects:
        if o.type == "ARMATURE" and "head" in o.pose.bones:
            arm = o; break
    modos = {pb.name: pb.rotation_mode for pb in arm.pose.bones} if arm else {}

    lim_corpus = limites_corpus(cfg) if cfg.get("FACIAL_CLAMP", "corpus") == "corpus" else None
    if lim_corpus is not None:
        print(f"clamp 'corpus': limites globais de {len(lim_corpus)} controles CTRL_*")
    sel = cfg.get("TRANSF_ALVOS", "todas")
    suf = cfg["TRANSF_SUFIXO_SLOT"]
    passo = max(1, int(cfg.get("PASSO_FRAMES", 1)))
    feitas = []
    ausentes = []

    for chave, rotulo in sorted(cfg["MAPA_ACTIONS"].items()):
        tema_tam, _, tipo = rotulo.rpartition("_")
        if canon_tipo(tipo) != tipo_af:          # v4.2: comparacao canonica
            continue
        if sel != "todas" and tema_tam not in sel:
            continue

        nome_act, act = resolver_action(chave, rotulo, cfg)   # v4.2
        if act is None:
            ausentes.append(chave)
            print("  action ausente:", chave)
            continue

        nome_novo = f"{nome_act}_{suf}"
        antiga = bpy.data.actions.get(nome_novo)
        if antiga is not None:
            bpy.data.actions.remove(antiga)
        novo = act.copy()
        novo.name = nome_novo
        # v4.3: sem fake user a copia fica com zero usuarios e o Blender NAO
        # a grava no .blend — as sinteticas sumiriam ao reabrir o arquivo.
        novo.use_fake_user = True

        slot, cb = channelbag_object(novo)
        if cb is None:
            print("  sem slot OBJECT:", nome_act); continue
        f0, f1 = frame_range(novo, cb)
        frames = np.arange(f0, f1 + 0.5, passo, dtype=float)
        fases = (frames - f0) / max(f1 - f0, 1e-9)
        idx = indexar(cb)

        nc = sum(aplicar_corpo(idx, cb, bn, idx[bn], tpl_c[bn], frames, fases,
                               modos.get(bn, "QUATERNION"), cfg)
                 for bn in tpl_c if bn in idx)
        nf = sum(aplicar_face(cb, bn, idx[bn], tpl_f[bn], frames, fases, cfg, lim_corpus)
                 for bn in tpl_f if bn in idx)
        feitas.append(novo)
        print(f"  {nome_novo:32s} ({rotulo} -> {alvo_arq})  frames {f0:.0f}-{f1:.0f}  corpo {nc}  face {nf}")

    if ausentes:
        print(f"\n{len(ausentes)} actions do mapa nao foram encontradas no .blend:")
        for nome in ausentes:
            print("   ", nome)

    if not feitas:
        print("nada gerado — confira TIPO_AF, TRANSF_ALVOS e MAPA_ACTIONS.")
        return
    if cfg.get("TRANSF_ATRIBUIR_AO_RIG", True) and arm is not None:
        ad = arm.animation_data or arm.animation_data_create()
        ad.action = feitas[0]
        s, _ = channelbag_object(feitas[0])
        if s is not None:
            ad.action_slot = s
        print(f"\natribuida ao rig para conferencia: {feitas[0].name}")
    print(f"\n{len(feitas)} actions sinteticas geradas. Salve o arquivo para manter.")


main()
