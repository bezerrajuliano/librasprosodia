bl_info = {
    "name": "Prosodia - Cortar Frames da Action",
    "author": "Juliano Bezerra",
    "version": (1, 0),
    "blender": (4, 4, 0),
    "location": "3D Viewport > Sidebar (N) > Prosodia",
    "description": "Apaga um intervalo de frames da action ativa do rig, em todos os slots (bones + shape keys)",
    "category": "Animation",
}

import bpy
import math
import numpy as np


# ==========================================================================
# NUCLEO
# ==========================================================================
def channelbags(act):
    """[(slot, channelbag)] de todos os slots da action que tem canais."""
    out = []
    for slot in act.slots:
        for layer in act.layers:
            for strip in layer.strips:
                cb = strip.channelbag(slot)
                if cb is not None:
                    out.append((slot, cb))
    return out


def extensao(cb):
    lo, hi, n = math.inf, -math.inf, 0
    for fc in cb.fcurves:
        k = len(fc.keyframe_points)
        if k == 0:
            continue
        r = fc.range()
        lo, hi = min(lo, r[0]), max(hi, r[1])
        n += k
    return (lo, hi, n) if n else None


def apagar_intervalo_fcurve(fc, f_ini, f_fim, fechar):
    """Remove keys em [f_ini, f_fim] e, se fechar, desloca os seguintes."""
    n = len(fc.keyframe_points)
    if n == 0:
        return 0
    co = np.empty(n * 2)
    fc.keyframe_points.foreach_get("co", co)
    frames = co[0::2]
    dentro = np.nonzero((frames >= f_ini - 0.5) & (frames <= f_fim + 0.5))[0]
    for i in reversed(dentro.tolist()):
        fc.keyframe_points.remove(fc.keyframe_points[i], fast=True)

    if fechar:
        delta = float(f_fim - f_ini + 1)
        n2 = len(fc.keyframe_points)
        if n2:
            for attr in ("co", "handle_left", "handle_right"):
                buf = np.empty(n2 * 2)
                fc.keyframe_points.foreach_get(attr, buf)
                fr = buf[0::2]              # view — altera buf no lugar
                fr[fr > f_fim] -= delta
                fc.keyframe_points.foreach_set(attr, buf)
    fc.update()
    return len(dentro)


def apagar_intervalo_action(act, f_ini, f_fim, fechar=True, so_object=False):
    total, slots = 0, 0
    for slot, cb in channelbags(act):
        if so_object and slot.target_id_type != "OBJECT":
            continue
        slots += 1
        for fc in cb.fcurves:
            total += apagar_intervalo_fcurve(fc, f_ini, f_fim, fechar)

    # se a action tem Manual Frame Range, encolhe junto para nao ficar defasado
    if fechar and act.use_frame_range:
        s, e = act.frame_range
        delta = f_fim - f_ini + 1
        if e > f_fim:
            e -= delta
        if s > f_fim:
            s -= delta
        act.frame_range = (s, max(s, e))
    return total, slots


# ==========================================================================
# PROPRIEDADES
# ==========================================================================
class ProsodiaCorteProps(bpy.types.PropertyGroup):
    frame_ini: bpy.props.IntProperty(
        name="Inicio", default=1, min=0,
        description="Primeiro frame a apagar (inclusive)")
    frame_fim: bpy.props.IntProperty(
        name="Fim", default=1, min=0,
        description="Ultimo frame a apagar (inclusive)")
    fechar_lacuna: bpy.props.BoolProperty(
        name="Fechar lacuna", default=True,
        description="Desloca os keyframes posteriores para tras, para nao sobrar buraco")
    so_object: bpy.props.BoolProperty(
        name="So slot Object (bones)", default=False,
        description="Se marcado, nao mexe nos slots de shape key. Normalmente deixe DESMARCADO")
    resumo: bpy.props.StringProperty(name="Resumo", default="")


# ==========================================================================
# OPERADORES
# ==========================================================================
def action_ativa(context):
    ob = context.active_object
    if ob is None or ob.type != "ARMATURE" or ob.animation_data is None:
        return None
    return ob.animation_data.action


class PROSODIA_OT_inspecionar(bpy.types.Operator):
    bl_idname = "prosodia.inspecionar_action"
    bl_label = "Ver extensao por slot"
    bl_description = "Mostra ate que frame vao os keyframes de cada slot da action ativa"

    def execute(self, context):
        act = action_ativa(context)
        if act is None:
            self.report({"WARNING"}, "Selecione o rig (armature) com uma action atribuida")
            return {"CANCELLED"}
        linhas = []
        for slot, cb in channelbags(act):
            e = extensao(cb)
            if e is None:
                continue
            nome = slot.name_display if hasattr(slot, "name_display") else slot.name
            linhas.append("{} [{}]: {:.0f}-{:.0f} ({} keys)".format(
                nome, slot.target_id_type, e[0], e[1], e[2]))
        texto = "  |  ".join(linhas) if linhas else "(sem keyframes)"
        context.scene.prosodia_corte.resumo = texto
        print("\n" + act.name)
        for l in linhas:
            print("   " + l)
        self.report({"INFO"}, texto[:200])
        return {"FINISHED"}


class PROSODIA_OT_usar_frame_atual(bpy.types.Operator):
    bl_idname = "prosodia.usar_frame_atual"
    bl_label = "Usar frame atual"
    campo: bpy.props.EnumProperty(items=[("INI", "Inicio", ""), ("FIM", "Fim", "")])

    def execute(self, context):
        p = context.scene.prosodia_corte
        if self.campo == "INI":
            p.frame_ini = context.scene.frame_current
        else:
            p.frame_fim = context.scene.frame_current
        return {"FINISHED"}


class PROSODIA_OT_apagar(bpy.types.Operator):
    bl_idname = "prosodia.apagar_frames"
    bl_label = "Apagar frames"
    bl_description = "Apaga o intervalo em todos os slots da action ativa (Ctrl+Z desfaz)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        act = action_ativa(context)
        if act is None:
            self.report({"WARNING"}, "Selecione o rig (armature) com uma action atribuida")
            return {"CANCELLED"}
        p = context.scene.prosodia_corte
        if p.frame_fim < p.frame_ini:
            self.report({"ERROR"}, "Fim menor que inicio")
            return {"CANCELLED"}
        if act.name.endswith("_sint"):
            self.report({"WARNING"}, "A action ativa e uma _sint (copia gerada). Confira se e isso mesmo.")

        total, slots = apagar_intervalo_action(
            act, p.frame_ini, p.frame_fim, fechar=p.fechar_lacuna, so_object=p.so_object)
        fr = act.frame_range
        msg = "{}: {} keys removidos em {} slots ({}-{}). Range agora {:.0f}-{:.0f}".format(
            act.name, total, slots, p.frame_ini, p.frame_fim, fr[0], fr[1])
        print(msg)
        self.report({"INFO"}, msg)
        # atualiza o resumo no painel
        bpy.ops.prosodia.inspecionar_action()
        return {"FINISHED"}


# ==========================================================================
# PAINEL
# ==========================================================================
class PROSODIA_PT_corte(bpy.types.Panel):
    bl_label = "Cortar frames da Action"
    bl_idname = "PROSODIA_PT_corte"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Prosodia"

    def draw(self, context):
        lay = self.layout
        p = context.scene.prosodia_corte
        act = action_ativa(context)

        box = lay.box()
        if act is None:
            box.label(text="Selecione o rig com uma action", icon="ERROR")
        else:
            fr = act.frame_range
            box.label(text=act.name, icon="ACTION")
            box.label(text="Range atual: {:.0f} - {:.0f}".format(fr[0], fr[1]))
            if act.use_frame_range:
                box.label(text="(Manual Frame Range ligado)", icon="INFO")
            box.operator("prosodia.inspecionar_action", icon="VIEWZOOM")
            if p.resumo:
                for parte in p.resumo.split("  |  "):
                    box.label(text=parte)

        col = lay.column(align=True)
        row = col.row(align=True)
        row.prop(p, "frame_ini")
        row.operator("prosodia.usar_frame_atual", text="", icon="TIME").campo = "INI"
        row = col.row(align=True)
        row.prop(p, "frame_fim")
        row.operator("prosodia.usar_frame_atual", text="", icon="TIME").campo = "FIM"

        lay.prop(p, "fechar_lacuna")
        lay.prop(p, "so_object")

        row = lay.row()
        row.scale_y = 1.6
        row.enabled = act is not None
        row.operator("prosodia.apagar_frames", icon="TRASH")


# ==========================================================================
# REGISTRO
# ==========================================================================
classes = (
    ProsodiaCorteProps,
    PROSODIA_OT_inspecionar,
    PROSODIA_OT_usar_frame_atual,
    PROSODIA_OT_apagar,
    PROSODIA_PT_corte,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.prosodia_corte = bpy.props.PointerProperty(type=ProsodiaCorteProps)


def unregister():
    del bpy.types.Scene.prosodia_corte
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    # permite rodar direto no Text Editor (Alt+P) sem instalar
    try:
        unregister()
    except Exception:
        pass
    register()
