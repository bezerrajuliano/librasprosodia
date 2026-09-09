"""
===========================================================================
 gerar_template_facial.py  (v4.3 — corpus 9 temas, Blender 5.2, INT e NEG, split-half)
===========================================================================
Le TODAS as actions juliano_NN (slot OBJECT = Rigify), identifica
tema/tamanho/tipo via "MAPA_ACTIONS" do config_prosodia.json e, para cada
controlador facial CTRL_* (canal .location), calcula:

    delta(t) = loc_alvo(t) - loc_af(t)     (fase normalizada, N_FASE pts)
    -> eixo dominante, corr_forma intra-tema, inter-temas, direcional,
       amplitude media. Simetria bilateral: pares CTRL_L_x / CTRL_R_x sao
       avaliados juntos; se um lado aprova, o outro herda a curva
       (FACIAL_SIMETRIA_FORCADA).

Visemas (CC_Base_Body/Tongue Key), tongue, jaw e boca sao conteudo lexical:
ficam fora salvo FACIAL_INCLUIR_ARTICULACAO = true.

Saidas ao lado do .blend (uma dupla por alvo):
    template_facial_int.csv    estatisticas_facial_int.csv
    template_facial_neg.csv    estatisticas_facial_neg.csv

Schema do template:
    alvo, bone, regiao, dom_eixo, corr_intra, corr_inter, direcional,
    amplitude, origem_simetria, fase_idx, fase, dx, dy, dz

Mesmo config_prosodia.json do script corporal (limiares unificados).
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

# ==========================================================================
# CONFIG PADRAO  (identico ao corporal — arquivo unico)
# ==========================================================================
CONFIG_PADRAO = {
    "N_FASE": 101,
    "ORDEM_PAR": "ALVO_MENOS_AF",
    "TIPO_AF": "1",
    "TIPOS_ALVO": {"2": "int"},   # acrescente "3": "neg" na 2a fase
    "TAMANHOS": ["curta", "normal", "longa"],
    "PADRAO_ACTION": "juliano_{:02d}",
    "ORDEM_TEMAS": ["viagem", "comida", "estudo", "compra", "passeio",
                    "vestido", "livro", "cafe", "futebol"],
    "MAPA_ACTIONS": {},
    "CORR_MINIMA": 0.50,
    "DIRECIONAL_MIN": 0.80,
    # Quais metricas funcionam como PORTAO de aprovacao. Opcoes:
    #   "intra"  corr entre tamanhos do mesmo tema (frases lexicalmente
    #            distintas -> mede variacao lexical, nao prosodia; so informativo)
    #   "inter"  corr entre curvas medias dos temas
    #   "split"  confiabilidade split-half: corr entre medias de duas metades
    #            aleatorias dos pares (SPLIT_REPETICOES sorteios)
    #   "dir"    consistencia direcional
    "CRITERIOS_APROVACAO": ["inter", "split", "dir"],
    "SPLIT_REPETICOES": 200,
    "ARMATURE": "",
    "CORPO_TERMOS_BONES": ["neck", "head", "spine_fk", "chest", "torso",
                           "tweak_spine", "shoulder", "hips"],
    "CORPO_PREFIXOS_EXCLUIR": ["CTRL_", "DEF-", "MCH-", "ORG-", "VIS_"],
    "CORPO_MIN_ANGULO_GRAUS": 2.0,
    "FACIAL_PREFIXO_CTRL": "CTRL_",
    "FACIAL_MIN_AMPLITUDE": 0.010,
    "FACIAL_INCLUIR_ARTICULACAO": False,
    "FACIAL_TERMOS_ARTICULACAO": ["jaw", "mouth", "lip", "corner", "tongue",
                                  "chin", "teeth", "neck_", "ear"],
    "FACIAL_SIMETRIA_FORCADA": True,
    # "independente": cada lado aprovado mantem sua propria curva (pode gerar
    #   timing assimetrico L/R na sintese); "media": quando os DOIS lados
    #   aprovam, ambos recebem a media das duas curvas — franzido simetrico.
    "FACIAL_SIMETRIA_MODO": "media",
    # Alinha os picos dos pares (deslocamento de fase por correlacao cruzada,
    # ate +-FACIAL_ALINHAR_MAX de fase) antes de tirar a media — recupera a
    # amplitude que a media "crua" dilui quando o timing varia entre frases.
    "FACIAL_ALINHAR_PICOS": True,
    "FACIAL_ALINHAR_MAX": 0.12,
}

ARQ_CONFIG = "config_prosodia.json"


def base_dir():
    b = bpy.path.abspath("//")
    if not b:
        raise RuntimeError("Salve o .blend antes de rodar.")
    return b


def gerar_mapa(cfg):
    mapa = {}
    tams, temas = cfg["TAMANHOS"], cfg["ORDEM_TEMAS"]
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
# FCURVES
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


RE_LOC = re.compile(r'pose\.bones\["(?P<bone>[^"]+)"\]\.location$')


def locs_ctrl(cb, frames, cfg):
    """{bone_CTRL: array (N,3)} das locations."""
    fcs = {}
    for fc in cb.fcurves:
        m = RE_LOC.match(fc.data_path)
        if m and m.group("bone").startswith(cfg["FACIAL_PREFIXO_CTRL"]):
            fcs.setdefault(m.group("bone"), {})[fc.array_index] = fc
    out = {}
    for bone, comp in fcs.items():
        a = np.zeros((len(frames), 3))
        for i in range(3):
            if i in comp:
                a[:, i] = [comp[i].evaluate(f) for f in frames]
        out[bone] = a
    return out


# ==========================================================================
# SIMETRIA / REGIAO
# ==========================================================================
RE_LADO = re.compile(r"^CTRL_(?P<lado>[LRC])_(?P<resto>.+)$")


def lado_base(nome):
    m = RE_LADO.match(nome)
    if not m:
        return None, nome
    return m.group("lado"), "CTRL_@_" + m.group("resto")


def regiao(nome):
    n = nome.lower()
    for r in ("brow", "eye", "nose", "jaw", "mouth", "tongue", "teeth",
              "neck", "ear", "head"):
        if r in n:
            return r
    return "outro"


def eh_articulacao(nome, cfg):
    n = nome.lower()
    return any(t.lower() in n for t in cfg["FACIAL_TERMOS_ARTICULACAO"])


# ==========================================================================
# METRICAS
# ==========================================================================
def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def split_half(curvas, n_rep, seed=0):
    """Corr media entre as curvas-media de duas metades aleatorias dos pares."""
    curvas = np.stack(curvas)
    n = len(curvas)
    if n < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        p = rng.permutation(n); h = n // 2
        vals.append(pearson(curvas[p[:h]].mean(axis=0), curvas[p[h:]].mean(axis=0)))
    return float(np.mean(vals))


def media_pares(curvas):
    if len(curvas) < 2:
        return float("nan")
    return float(np.mean([pearson(curvas[i], curvas[j])
                          for i in range(len(curvas)) for j in range(i + 1, len(curvas))]))


def _desloca(c, s_pts):
    """Desloca a curva s_pts pontos de fase, segurando as bordas."""
    n = len(c); i = np.arange(n, dtype=float) - s_pts
    return np.interp(np.clip(i, 0, n - 1), np.arange(n), c)


def alinhar_picos(todas, dom, max_fase, passos=2):
    """todas (P,N,3): desloca cada par p/ maximizar corr com a media no eixo dom."""
    n = todas.shape[1]
    smax = max(1, int(round(max_fase * (n - 1))))
    desloc = np.zeros(len(todas), dtype=int)
    for _ in range(passos):
        ref = todas[:, :, dom].mean(axis=0)
        ref = ref - ref.mean()
        for p in range(len(todas)):
            c = todas[p, :, dom]
            melhor, arg = -2.0, 0
            for sh in range(-smax, smax + 1):
                d = _desloca(c, sh); d = d - d.mean()
                den = np.sqrt((d * d).sum() * (ref * ref).sum())
                r = float((d * ref).sum() / den) if den > 1e-12 else 0.0
                if r > melhor:
                    melhor, arg = r, sh
            if arg:
                for e in range(3):
                    todas[p, :, e] = _desloca(todas[p, :, e], arg)
                desloc[p] += arg
    return todas, desloc


def avaliar(deltas, cfg):
    todas = np.stack(list(deltas.values()))
    media = todas.mean(axis=0)
    dom = int(np.argmax(np.abs(media).max(axis=0)))
    desloc = np.zeros(len(todas), dtype=int)
    if cfg.get("FACIAL_ALINHAR_PICOS", True) and len(todas) >= 4:
        todas, desloc = alinhar_picos(todas.copy(), dom, cfg.get("FACIAL_ALINHAR_MAX", 0.12))
        deltas = {k: todas[i] for i, k in enumerate(deltas)}
        media = todas.mean(axis=0)
        dom = int(np.argmax(np.abs(media).max(axis=0)))
    pico_idx = int(np.argmax(np.abs(media[:, dom])))
    sinal = np.sign(media[pico_idx, dom]) or 1.0
    por_tema = {}
    for (tema, tam), d in deltas.items():
        por_tema.setdefault(tema, []).append(d[:, dom])
    intra = [media_pares(c) for c in por_tema.values() if len(c) >= 2]
    corr_intra = float(np.mean(intra)) if intra else float("nan")
    corr_inter = media_pares([np.mean(c, axis=0) for c in por_tema.values()])
    corr_split = split_half([d[:, dom] for d in deltas.values()], cfg["SPLIT_REPETICOES"])
    direcional = float(np.mean([np.sign(d[pico_idx, dom]) == sinal for d in deltas.values()]))
    amplitude = float(np.abs(media[pico_idx, dom]))
    crit = cfg["CRITERIOS_APROVACAO"]
    cmin = cfg.get("FACIAL_CORR_MINIMA", cfg["CORR_MINIMA"])   # limiar proprio da face, se existir
    ok = amplitude >= cfg["FACIAL_MIN_AMPLITUDE"]
    if "intra" in crit: ok = ok and corr_intra >= cmin
    if "inter" in crit: ok = ok and corr_inter >= cmin
    if "split" in crit: ok = ok and corr_split >= cmin
    if "dir"   in crit: ok = ok and direcional >= cfg["DIRECIONAL_MIN"]
    return dict(media=media, dom_eixo="XYZ"[dom], corr_intra=corr_intra,
                corr_inter=corr_inter, corr_split=corr_split, direcional=direcional, amplitude=amplitude,
                fase_pico=pico_idx / (len(media) - 1), n_pares=len(deltas),
                desloc_medio=float(np.mean(np.abs(desloc))) / (len(media) - 1), aprovado=ok)


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    cfg = carregar_config()
    N = cfg["N_FASE"]
    fases = np.linspace(0.0, 1.0, N)

    unidades = {}
    for act in bpy.data.actions:
        rot = cfg["MAPA_ACTIONS"].get(act.name)
        if not rot:
            continue
        u = parse_unidade(rot)
        if not u:
            print("  rotulo invalido:", act.name, "->", rot); continue
        slot, cb = channelbag_object(act)
        if cb is None:
            print("  sem slot OBJECT:", act.name); continue
        f0, f1 = frame_range(act, cb)
        frames = f0 + fases * (f1 - f0)
        unidades[u] = locs_ctrl(cb, frames, cfg)
        print(f"  {act.name:12s} -> {rot:18s} frames {f0:.0f}-{f1:.0f}  CTRL {len(unidades[u])}")

    if not unidades:
        raise RuntimeError("Nenhuma action mapeada. Confira MAPA_ACTIONS.")

    for tipo_alvo, nome_alvo in cfg["TIPOS_ALVO"].items():
        pares = [(t, s) for (t, s, k) in unidades
                 if k == cfg["TIPO_AF"] and (t, s, tipo_alvo) in unidades]
        print(f"\n=== ALVO {nome_alvo.upper()} (tipo {tipo_alvo}): {len(pares)} pares ===")
        if not pares:
            continue

        bones = set()
        for t, s in pares:
            bones |= set(unidades[(t, s, cfg["TIPO_AF"])]) & set(unidades[(t, s, tipo_alvo)])
        excl_cfg = cfg.get("FACIAL_PARES_EXCLUIDOS", [])
        if excl_cfg:
            fora = {b for b in bones if any(t.lower() in b.lower() for t in excl_cfg)}
            print(f"  excluidos por FACIAL_PARES_EXCLUIDOS: {sorted(fora)}")
            bones -= fora
        if not cfg["FACIAL_INCLUIR_ARTICULACAO"]:
            excl = {b for b in bones if eh_articulacao(b, cfg)}
            print(f"  articulacao excluida: {len(excl)} canais")
            bones -= excl

        # avaliacao individual
        res = {}
        for bone in sorted(bones):
            deltas = {}
            for t, s in pares:
                a = unidades[(t, s, cfg["TIPO_AF"])].get(bone)
                b = unidades[(t, s, tipo_alvo)].get(bone)
                if a is not None and b is not None:
                    deltas[(t, s)] = b - a
            if len(deltas) >= 2:
                res[bone] = avaliar(deltas, cfg)

        # simetria bilateral
        grupos = {}
        for bone in res:
            lado, base = lado_base(bone)
            grupos.setdefault(base, {})[lado] = bone
        origem = {}   # bone -> bone de onde herdou a curva
        for base, lados in grupos.items():
            L, R = lados.get("L"), lados.get("R")
            if not (L and R):
                continue
            okL, okR = res[L]["aprovado"], res[R]["aprovado"]
            if cfg["FACIAL_SIMETRIA_FORCADA"] and okL != okR:
                bom, ruim = (L, R) if okL else (R, L)
                m = dict(res[bom]); m["aprovado"] = True
                res[ruim] = m
                origem[ruim] = bom
            elif okL and okR and cfg.get("FACIAL_SIMETRIA_MODO", "independente") == "media":
                media_LR = 0.5 * (res[L]["media"] + res[R]["media"])
                for lado in (L, R):
                    m = dict(res[lado]); m["media"] = media_LR
                    dom = int(np.argmax(np.abs(media_LR).max(axis=0)))
                    pk = int(np.argmax(np.abs(media_LR[:, dom])))
                    m["dom_eixo"] = "XYZ"[dom]
                    m["amplitude"] = float(np.abs(media_LR[pk, dom]))
                    m["fase_pico"] = pk / (len(media_LR) - 1)
                    res[lado] = m
                    origem[lado] = "media(L,R)"

        stats, template = [], []
        for bone, m in res.items():
            stats.append(dict(alvo=nome_alvo, bone=bone, regiao=regiao(bone),
                              n_pares=m["n_pares"], dom_eixo=m["dom_eixo"],
                              corr_intra=round(m["corr_intra"], 4),
                              corr_inter=round(m["corr_inter"], 4),
                              corr_split=round(m["corr_split"], 4),
                              direcional=round(m["direcional"], 3),
                              amplitude=round(m["amplitude"], 5),
                              fase_pico=round(m["fase_pico"], 3),
                              desloc_medio=round(m["desloc_medio"], 3),
                              origem_simetria=origem.get(bone, ""),
                              aprovado=int(m["aprovado"])))
            if m["aprovado"]:
                for k in range(N):
                    template.append(dict(alvo=nome_alvo, bone=bone, regiao=regiao(bone),
                                         dom_eixo=m["dom_eixo"],
                                         corr_intra=round(m["corr_intra"], 4),
                                         corr_inter=round(m["corr_inter"], 4),
                                         corr_split=round(m["corr_split"], 4),
                                         direcional=round(m["direcional"], 3),
                                         amplitude=round(m["amplitude"], 5),
                                         origem_simetria=origem.get(bone, ""),
                                         fase_idx=k, fase=round(fases[k], 4),
                                         dx=round(float(m["media"][k, 0]), 6),
                                         dy=round(float(m["media"][k, 1]), 6),
                                         dz=round(float(m["media"][k, 2]), 6)))

        stats.sort(key=lambda r: (-r["aprovado"], -r["amplitude"]))
        b = base_dir()
        for nome, linhas in ((f"estatisticas_facial_{nome_alvo}.csv", stats),
                             (f"template_facial_{nome_alvo}.csv", template)):
            if not linhas:
                print("  (vazio)", nome); continue
            with open(os.path.join(b, nome), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
                w.writeheader(); w.writerows(linhas)
            print("  gravado:", nome, f"({len(linhas)} linhas)")

        print(f"\n  {'bone':32s} eixo  intra   inter   split   dir   ampl    fase  ok")
        for r in stats[:40]:
            print(f"  {r['bone']:32s}  {r['dom_eixo']}   {r['corr_intra']:6.3f}  "
                  f"{r['corr_inter']:6.3f}  {r['corr_split']:6.3f}  {r['direcional']:4.2f}  {r['amplitude']:.4f}  "
                  f"{r['fase_pico']:4.2f}  {'*' if r['aprovado'] else ''}"
                  f"{' (sim<-' + r['origem_simetria'] + ')' if r['origem_simetria'] else ''}")
        print(f"  aprovados: {sum(r['aprovado'] for r in stats)} de {len(stats)}")


main()
