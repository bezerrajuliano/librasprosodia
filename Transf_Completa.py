"""
===========================================================================
 transformacao_completa_slot.py  (v4.5 — protecao total + resolvedor definitivo de nomes)
===========================================================================
Sintetiza a versao prosodica (INT por padrao) de cada frase AFIRMATIVA
aplicando os templates gerados:

    corpo : q_sint(f) = q_delta(fase)^k . q_af(f)     (k = intensidade)
    face  : loc_sint(f) = loc_af(f) + k * delta(fase)  nos CTRL_* aprovados

Para cada afirmativa cria uma COPIA "<action>_<sufixo>" e faz o bake nela.
As originais NUNCA sao modificadas nem removidas.

--- v4.4: PROTECOES CONTRA PERDA DE ACTIONS ------------------------------
O Blender APAGA ao salvar toda action com zero usuarios e sem fake user
(o escudo F) — foi assim que as afirmativas se perderam. Agora:
  P1. Antes de qualquer coisa, TODAS as actions do corpus resolvidas no
      .blend recebem use_fake_user=True (af, int e neg — nao so as usadas).
  P2. bpy.data.actions.remove() so pode atingir nomes terminados no sufixo
      sintetico e diferentes da action de origem; qualquer outro alvo
      aborta o script com erro, por construcao.
  P3. Checagem de completude: se o numero de afirmativas resolvidas for
      menor que o esperado pelo MAPA_ACTIONS, o script lista as ausentes e
      ABORTA sem criar nada (rode com TRANSF_IGNORAR_AUSENTES=true no
      config para prosseguir mesmo assim, p.ex. num teste parcial).
  P4. Auditoria final: qualquer action que ainda sumiria ao salvar (0
      usuarios, sem F) e listada em alerta antes do fim.
Copias sinteticas continuam com use_fake_user=True (v4.3).

Mantidas da v4.2/4.3: resolver_action() (aceita juliano_NN, tema_tam_N ou
tema_tam_extenso, em qualquer combinacao entre config e .blend),
canon_tipo(), log dos valores efetivos do config (o JSON tem precedencia
sobre PADRAO_TRANSF), clamp "corpus" com resolvedor.

Chaves de controle (config_prosodia.json):
  TRANSF_ALVO           "int" / "interrogativa" (ou "neg" na 2a fase)
  TRANSF_ALVOS          "todas" ou lista ["viagem_curta", ...]
  TRANSF_INTENSIDADE_CABECA / _TRONCO / _FACIAL   escalas dos deltas
  TRANSF_TERMOS_CABECA  bones tratados como cabeca (demais = tronco)
  TRANSF_IGNORAR_AUSENTES  false (padrao): aborta se faltar afirmativa
  FACIAL_SOMENTE_EIXO_DOMINANTE / FACIAL_CLAMP / FACIAL_CLAMP_MARGEM
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

# ==========================================================================
# RESOLVEDOR DE NOMES DE ACTION  (definitivo — aceita qualquer formato)
# ==========================================================================
# Converte QUALQUER nome de action no rotulo canonico "tema_tamanho_N",
# com N = 1 afirmativa, 2 interrogativa, 3 negativa. Aceita:
#   comida_curta_afirmacao / _afirmativa / _af / _1   (idem interrogativa,
#   negativa, com ou sem acento, maiuscula ou espaco)
#   juliano_07 / juliano_7  (legado: convertido pela posicao no corpus)
#   qualquer chave ou valor presente no MAPA_ACTIONS do config
# Copias sinteticas (sufixo _sint) sao ignoradas de proposito.
import re as _re_mod
import unicodedata as _ud

_TIPOS = {
    "1": "1", "af": "1", "afir": "1", "afirm": "1",
    "afirmacao": "1", "afirmativa": "1", "afirmativo": "1", "afirmacoes": "1",
    "2": "2", "int": "2", "interr": "2", "interrog": "2",
    "interrogacao": "2", "interrogativa": "2", "interrogativo": "2",
    "3": "3", "neg": "3", "negacao": "3", "negativa": "3", "negativo": "3",
}
_RE_LEGADO = _re_mod.compile(r"^juliano_?0*(\d+)$")


def _norm(s):
    s = _ud.normalize("NFKD", str(s))
    s = "".join(c for c in s if not _ud.combining(c))
    return s.lower().replace(" ", "_").replace("-", "_").strip("_ ")


def canon_tipo_txt(t):
    """'afirmacao', 'af', '1' -> '1'  (idem 2 e 3). None se nao reconhecer."""
    return _TIPOS.get(_norm(t))


def _parse_rotulo(nome, cfg):
    partes = _norm(nome).split("_")
    if len(partes) < 3:
        return None
    tipo = canon_tipo_txt(partes[-1])
    if tipo is None:
        return None
    temas = [_norm(t) for t in cfg["ORDEM_TEMAS"]]
    tams = [_norm(t) for t in cfg["TAMANHOS"]]
    tam, tema = partes[-2], "_".join(partes[:-2])
    if tema in temas and tam in tams:
        return "{}_{}_{}".format(cfg["ORDEM_TEMAS"][temas.index(tema)],
                                 cfg["TAMANHOS"][tams.index(tam)], tipo)
    return None


def rotulo_da_action(nome, cfg):
    n = _norm(nome)
    suf = _norm(cfg.get("TRANSF_SUFIXO_SLOT", "sint"))
    if n.endswith("_" + suf):
        return None
    r = _parse_rotulo(n, cfg)
    if r:
        return r
    for chave, rot in cfg.get("MAPA_ACTIONS", {}).items():
        if _norm(chave) == n or _norm(rot) == n:
            return _parse_rotulo(rot, cfg) or rot
    m = _RE_LEGADO.match(n)
    if m:
        i = int(m.group(1)) - 1
        temas, tams = cfg["ORDEM_TEMAS"], cfg["TAMANHOS"]
        if 0 <= i < len(temas) * len(tams) * 3:
            passo = len(tams) * 3
            return "{}_{}_{}".format(temas[i // passo], tams[(i % passo) // 3], i % 3 + 1)
    return None


def indice_rotulos(cfg):
    """{rotulo_canonico: action} varrendo todas as actions do .blend."""
    idx = {}
    for a in bpy.data.actions:
        r = rotulo_da_action(a.name, cfg)
        if r:
            idx[r] = a
    return idx


def achar_action(nome):
    if nome is None:
        return None
    act = bpy.data.actions.get(nome)
    if act is not None:
        return act
    alvo = _norm(nome)
    for a in bpy.data.actions:
        if _norm(a.name) == alvo:
            return a
    return None
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

CANON_TIPO = {
    "1": "afirmativa", "af": "afirmativa", "afir": "afirmativa",
    "afirmativa": "afirmativa",
    "2": "interrogativa", "int": "interrogativa",
    "interrogativa": "interrogativa",
    "3": "negativa", "neg": "negativa", "negativa": "negativa",
}

NUM_PARA_EXTENSO = {"1": "afirmativa", "2": "interrogativa", "3": "negativa"}


def canon_tipo(t):
    s = str(t).strip().lower()
    return CANON_TIPO.get(s, s)


def variantes_nome(nome):
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


RE_NUM_FIM = re.compile(r"^(?P<base>.*_)0*(?P<num>\d+)$")


def variantes_padding(nome):
    """juliano_01 <-> juliano_1 (e 3 digitos, por seguranca)."""
    m = RE_NUM_FIM.match(nome)
    if not m:
        return [nome]
    b, n = m.group("base"), int(m.group("num"))
    return [nome, f"{b}{n}", f"{b}{n:02d}", f"{b}{n:03d}"]


def resolver_action(chave, rotulo, cfg=None):
    candidatos = []
    for n in (chave, rotulo):
        for v0 in variantes_nome(n):
            for v in variantes_padding(v0):
                if v not in candidatos:
                    candidatos.append(v)
    if cfg is not None:
        pos = nome_posicional(cfg, rotulo)
        if pos:
            for v in variantes_padding(pos):
                if v not in candidatos:
                    candidatos.append(v)
    for nome in candidatos:
        act = bpy.data.actions.get(nome)
        if act is not None:
            return nome, act
    # ultimo recurso: comparacao sem maiusculas/minusculas
    minusc = {a.name.lower(): a for a in bpy.data.actions}
    for nome in candidatos:
        act = minusc.get(nome.lower())
        if act is not None:
            return act.name, act
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


def ler_template(caminho, comps):
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
    out = np.empty((len(fases), 3))
    for c in range(3):
        out[:, c] = np.interp(fases, tpl["fase"], tpl["val"][:, c])
    return out


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
    lim = {}
    achadas = 0
    for rotulo, act in sorted(indice_rotulos(cfg).items()):
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
    print(f"clamp 'corpus': {achadas} actions varridas")
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


# ==========================================================================
# v4.4 — PROTECOES
# ==========================================================================
def proteger_corpus(cfg):
    """P1: fake user em todas as actions do corpus presentes no .blend.

    Varre as actions do arquivo e classifica cada uma pelo rotulo canonico
    (aceita afirmacao/afirmativa/af/1 etc.), em vez de depender do formato
    de nome gravado no MAPA_ACTIONS."""
    resolvidas, ausentes, protegidas = {}, [], 0
    idx = indice_rotulos(cfg)
    esperados = ["{}_{}_{}".format(t, s, k) for t in cfg["ORDEM_TEMAS"]
                 for s in cfg["TAMANHOS"] for k in "123"]
    for rotulo in esperados:
        act = idx.get(rotulo)
        if act is None:
            ausentes.append(rotulo)
            continue
        resolvidas[rotulo] = (act.name, act)
        if not act.use_fake_user:
            act.use_fake_user = True
            protegidas += 1
    print(f"protecao P1: {len(resolvidas)}/{len(esperados)} actions do corpus "
          f"resolvidas; fake user ligado em {protegidas}")
    return resolvidas, ausentes


def remover_sintetica_antiga(nome_novo, suf, origem):
    """P2: remove APENAS uma copia sintetica antiga; qualquer outro alvo aborta."""
    antiga = bpy.data.actions.get(nome_novo)
    if antiga is None:
        return
    if not nome_novo.endswith("_" + suf):
        raise RuntimeError(f"P2: recusado remover '{nome_novo}' — nome nao termina em '_{suf}'.")
    if antiga == origem:
        raise RuntimeError(f"P2: recusado remover '{nome_novo}' — e a propria action de origem.")
    bpy.data.actions.remove(antiga)


def auditoria_final():
    """P4: alerta sobre qualquer action que o Blender apagaria ao salvar."""
    risco = [a.name for a in bpy.data.actions if a.users == 0 and not a.use_fake_user]
    if risco:
        print("\n" + "!" * 70)
        print("ALERTA P4: estas actions tem 0 usuarios e SEM fake user —")
        print("elas SUMIRAO no proximo salvamento se nada as referenciar:")
        for n in risco:
            print("   ", n)
        print("!" * 70)
    else:
        print("auditoria P4: nenhuma action em risco de purga ao salvar.")


def main():
    cfg = carregar_config()
    alvo = cfg.get("TRANSF_ALVO", "int")
    tipo_af = canon_tipo(cfg.get("TIPO_AF", "1"))
    b = base_dir()

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

    # P1 + P3 — proteger e checar completude ANTES de criar qualquer coisa
    resolvidas, ausentes = proteger_corpus(cfg)
    tipo_af_num = canon_tipo_txt(cfg.get("TIPO_AF", "1")) or "1"
    esperadas_af = ["{}_{}_{}".format(t, s_, tipo_af_num) for t in cfg["ORDEM_TEMAS"]
                    for s_ in cfg["TAMANHOS"]]
    af_resolvidas = [r for r in resolvidas if r.endswith("_" + tipo_af_num)]
    if len(af_resolvidas) < len(esperadas_af):
        faltam = sorted(set(esperadas_af) - set(af_resolvidas))
        print(f"\nP3: faltam {len(faltam)} afirmativas no .blend:")
        for f_ in faltam:
            print("   ", f_)
        nao_sint = [a.name for a in bpy.data.actions if "_sint" not in a.name.lower()]
        print(f"\nP3: para comparar, {len(nao_sint)} actions existentes no .blend (ate 30):")
        for n in sorted(nao_sint)[:30]:
            print("   ", n)
        if not cfg.get("TRANSF_IGNORAR_AUSENTES", False):
            raise RuntimeError("P3: corpus incompleto — nada foi criado. "
                               "(TRANSF_IGNORAR_AUSENTES=true no config para prosseguir)")

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
    sel = cfg.get("TRANSF_ALVOS", "todas")
    suf = cfg["TRANSF_SUFIXO_SLOT"]
    passo = max(1, int(cfg.get("PASSO_FRAMES", 1)))
    feitas = []

    for rotulo in sorted(af_resolvidas):
        tema_tam = rotulo.rpartition("_")[0]
        if sel != "todas" and tema_tam not in sel:
            continue
        nome_act, act = resolvidas[rotulo]

        nome_novo = f"{nome_act}_{suf}"
        remover_sintetica_antiga(nome_novo, suf, act)      # P2
        novo = act.copy()
        novo.name = nome_novo
        novo.use_fake_user = True                          # v4.3
        if novo.name != nome_novo:                         # colisao de nome
            raise RuntimeError(f"P2: '{nome_novo}' ainda existia; copia virou '{novo.name}'. "
                               "Limpe duplicatas e rode de novo.")

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
        print(f"  {nome_novo:36s} ({rotulo} -> {alvo_arq})  frames {f0:.0f}-{f1:.0f}  corpo {nc}  face {nf}")

    if not feitas:
        print("nada gerado — confira TIPO_AF, TRANSF_ALVOS e MAPA_ACTIONS.")
        auditoria_final()
        return
    if cfg.get("TRANSF_ATRIBUIR_AO_RIG", True) and arm is not None:
        ad = arm.animation_data or arm.animation_data_create()
        ad.action = feitas[0]
        s, _ = channelbag_object(feitas[0])
        if s is not None:
            ad.action_slot = s
        print(f"\natribuida ao rig para conferencia: {feitas[0].name}")

    auditoria_final()                                      # P4
    print(f"\n{len(feitas)} actions sinteticas geradas. Salve o arquivo para manter.")


main()
