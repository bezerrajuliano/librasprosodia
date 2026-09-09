"""
Renomeia as Actions do arquivo .blend conforme o mapa abaixo.

O sufixo numerico do nome de destino e convertido automaticamente:
    _1 -> _afirmativa
    _2 -> _interrogativa
    _3 -> _negativa

Ex.: juliano_01  ->  viagem_curta_afirmativa

Como usar:
    1. Abra o .blend no Blender.
    2. Va em Scripting > New, cole este arquivo e rode (Alt+P).
    3. Deixe DRY_RUN = True na primeira rodada para so conferir o relatorio
       no console (Window > Toggle System Console). Depois mude para False
       e rode de novo para aplicar de verdade.
"""

import bpy

# True  = apenas simula e imprime o relatorio (nao altera nada)
# False = aplica os novos nomes
DRY_RUN = False

SUFIXOS = {
    "1": "afirmativa",
    "2": "interrogativa",
    "3": "negativa",
}

MAPA = {
    "juliano_01": "viagem_curta_1",
    "juliano_02": "viagem_curta_2",
    "juliano_03": "viagem_curta_3",
    "juliano_04": "viagem_normal_1",
    "juliano_05": "viagem_normal_2",
    "juliano_06": "viagem_normal_3",
    "juliano_07": "viagem_longa_1",
    "juliano_08": "viagem_longa_2",
    "juliano_09": "viagem_longa_3",
    "juliano_10": "comida_curta_1",
    "juliano_11": "comida_curta_2",
    "juliano_12": "comida_curta_3",
    "juliano_13": "comida_normal_1",
    "juliano_14": "comida_normal_2",
    "juliano_15": "comida_normal_3",
    "juliano_16": "comida_longa_1",
    "juliano_17": "comida_longa_2",
    "juliano_18": "comida_longa_3",
    "juliano_19": "estudo_curta_1",
    "juliano_20": "estudo_curta_2",
    "juliano_21": "estudo_curta_3",
    "juliano_22": "estudo_normal_1",
    "juliano_23": "estudo_normal_2",
    "juliano_24": "estudo_normal_3",
    "juliano_25": "estudo_longa_1",
    "juliano_26": "estudo_longa_2",
    "juliano_27": "estudo_longa_3",
    "juliano_28": "compra_curta_1",
    "juliano_29": "compra_curta_2",
    "juliano_30": "compra_curta_3",
    "juliano_31": "compra_normal_1",
    "juliano_32": "compra_normal_2",
    "juliano_33": "compra_normal_3",
    "juliano_34": "compra_longa_1",
    "juliano_35": "compra_longa_2",
    "juliano_36": "compra_longa_3",
    "juliano_37": "passeio_curta_1",
    "juliano_38": "passeio_curta_2",
    "juliano_39": "passeio_curta_3",
    "juliano_40": "passeio_normal_1",
    "juliano_41": "passeio_normal_2",
    "juliano_42": "passeio_normal_3",
    "juliano_43": "passeio_longa_1",
    "juliano_44": "passeio_longa_2",
    "juliano_45": "passeio_longa_3",
    "juliano_46": "vestido_curta_1",
    "juliano_47": "vestido_curta_2",
    "juliano_48": "vestido_curta_3",
    "juliano_49": "vestido_normal_1",
    "juliano_50": "vestido_normal_2",
    "juliano_51": "vestido_normal_3",
    "juliano_52": "vestido_longa_1",
    "juliano_53": "vestido_longa_2",
    "juliano_54": "vestido_longa_3",
    "juliano_55": "livro_curta_1",
    "juliano_56": "livro_curta_2",
    "juliano_57": "livro_curta_3",
    "juliano_58": "livro_normal_1",
    "juliano_59": "livro_normal_2",
    "juliano_60": "livro_normal_3",
    "juliano_61": "livro_longa_1",
    "juliano_62": "livro_longa_2",
    "juliano_63": "livro_longa_3",
    "juliano_64": "cafe_curta_1",
    "juliano_65": "cafe_curta_2",
    "juliano_66": "cafe_curta_3",
    "juliano_67": "cafe_normal_1",
    "juliano_68": "cafe_normal_2",
    "juliano_69": "cafe_normal_3",
    "juliano_70": "cafe_longa_1",
    "juliano_71": "cafe_longa_2",
    "juliano_72": "cafe_longa_3",
    "juliano_73": "futebol_curta_1",
    "juliano_74": "futebol_curta_2",
    "juliano_75": "futebol_curta_3",
    "juliano_76": "futebol_normal_1",
    "juliano_77": "futebol_normal_2",
    "juliano_78": "futebol_normal_3",
    "juliano_79": "futebol_longa_1",
    "juliano_80": "futebol_longa_2",
    "juliano_81": "futebol_longa_3",
}


def converter_sufixo(destino):
    """viagem_curta_1 -> viagem_curta_afirmativa"""
    base, _, numero = destino.rpartition("_")
    if not base or numero not in SUFIXOS:
        return None
    return "{}_{}".format(base, SUFIXOS[numero])


def main():
    renomeadas = []
    ausentes = []
    conflitos = []
    invalidas = []

    for antigo, destino in sorted(MAPA.items()):
        novo = converter_sufixo(destino)
        if novo is None:
            invalidas.append((antigo, destino))
            continue

        action = bpy.data.actions.get(antigo)
        if action is None:
            ausentes.append(antigo)
            continue

        ocupado = bpy.data.actions.get(novo)
        if ocupado is not None and ocupado is not action:
            conflitos.append((antigo, novo))
            continue

        if not DRY_RUN:
            action.name = novo
        renomeadas.append((antigo, novo))

    print("")
    print("=" * 60)
    print("MODO: {}".format("SIMULACAO (nada foi alterado)" if DRY_RUN else "APLICADO"))
    print("=" * 60)

    for antigo, novo in renomeadas:
        print("  {:<14} -> {}".format(antigo, novo))

    if ausentes:
        print("")
        print("NAO ENCONTRADAS no arquivo ({}):".format(len(ausentes)))
        for nome in ausentes:
            print("  {}".format(nome))

    if conflitos:
        print("")
        print("CONFLITO - ja existe uma Action com o nome de destino ({}):".format(len(conflitos)))
        for antigo, novo in conflitos:
            print("  {} -> {}".format(antigo, novo))

    if invalidas:
        print("")
        print("SUFIXO INVALIDO no mapa ({}):".format(len(invalidas)))
        for antigo, destino in invalidas:
            print("  {} -> {}".format(antigo, destino))

    print("")
    print("Resumo: {} renomeadas | {} ausentes | {} conflitos | {} invalidas".format(
        len(renomeadas), len(ausentes), len(conflitos), len(invalidas)))
    print("Total de Actions no arquivo: {}".format(len(bpy.data.actions)))
    print("=" * 60)


main()
