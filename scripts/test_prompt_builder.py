from __future__ import annotations

from typing import Any, Dict
import numpy as np


def _safe_get(arr, idx, default=None):
    """Safely get arr[idx] returning default if out of range."""
    try:
        if idx < len(arr):
            return arr[idx]
    except Exception:
        pass
    return default


def _line_label(env, obs, line_id: int) -> str:
    """Return a unique label for a line using substation ids."""
    or_sub = int(obs.line_or_to_subid[line_id])
    ex_sub = int(obs.line_ex_to_subid[line_id])
    return f"{or_sub}_{ex_sub}_{line_id}"


def _format_substations(env, obs) -> str:
    """Return one table per substation listing elements and their busbars."""
    parts = [
        "Substations (elements and their assigned busbars):",
        "- sub_<id>: list of elements connected to the substation",
        "- busbar: global busbar id for each element in the same order",
        "  (busbars are numbered globally by combining substation id and local bus index)",
        "- line names use <sub_or>_<sub_ex>_<line_id> based on substation ids",
    ]
    n_sub = getattr(obs, "n_sub", 0)
    for sub_id in range(n_sub):
        try:
            mat = env.action_space.get_obj_substations(substation_id=sub_id)
        except Exception:
            continue
        elements = []
        buses = []
        for row in mat:
            topo_idx: int | None = None
            if row[env.action_space.LOA_COL] != -1:
                idx = int(row[env.action_space.LOA_COL])
                elements.append(f"load_{idx}")
                topo_idx = obs.load_pos_topo_vect[idx]
            elif row[env.action_space.GEN_COL] != -1:
                idx = int(row[env.action_space.GEN_COL])
                elements.append(f"gen_{idx}")
                topo_idx = obs.gen_pos_topo_vect[idx]
            elif row[env.action_space.LOR_COL] != -1:
                idx = int(row[env.action_space.LOR_COL])
                elements.append(_line_label(env, obs, idx))
                topo_idx = obs.line_or_pos_topo_vect[idx]
            elif row[env.action_space.LEX_COL] != -1:
                idx = int(row[env.action_space.LEX_COL])
                elements.append(_line_label(env, obs, idx))
                topo_idx = obs.line_ex_pos_topo_vect[idx]
            elif row[env.action_space.STORAGE_COL] != -1:
                idx = int(row[env.action_space.STORAGE_COL])
                elements.append(f"storage_{idx}")
                topo_idx = obs.storage_pos_topo_vect[idx]
            if topo_idx is None:
                continue
            local_bus = int(obs.topo_vect[topo_idx])
            buses.append(env.action_space.local_bus_to_global_int(local_bus, sub_id))
        if not elements:
            continue
        parts.append(f"sub_{sub_id}: | " + " | ".join(elements))
        parts.append("busbar: | " + " | ".join(str(b) for b in buses))
    return "\n".join(parts)


def _format_lines(env, obs) -> str:
    """Return a single table of line busbars and power flows."""
    obs_dict: Dict[str, Any] = obs.to_dict()
    lines_or = obs_dict.get("lines_or", {})
    lines_ex = obs_dict.get("lines_ex", {})
    rho = obs_dict.get("rho", [])
    overflow = obs_dict.get("timestep_overflow", [])
    protect = obs_dict.get("timestep_protection_engaged", [])
    maint = obs_dict.get("maintenance", {})
    maint_t = maint.get("time_next_maintenance", [])
    maint_d = maint.get("duration_next_maintenance", [])
    cooldown = obs_dict.get("cooldown", {}).get("line", [])

    caption = "Lines (connections between substations):"
    bullets = [
        "- line_name: label <sub_or>_<sub_ex>_<id> using substation ids",
        "- bus_or/bus_ex: global busbar ids at origin and extremity",
        "- p_or/q_or: active/reactive power at origin (MW/MVAr)",
        "- v_or/a_or: voltage (kV) and current (A) at origin",
        "- p_ex/q_ex: active/reactive power at extremity (MW/MVAr)",
        "- v_ex/a_ex: voltage (kV) and current (A) at extremity",
        "- rho: thermal loading ratio",
        "- overflow: 1 if line overflowed",
        "- protect: 1 if protection engaged",
        "- maint: time_remaining/duration of maintenance",
        "- cooldown: cooldown steps before reconnection",
    ]
    header = (
        "line_name | bus_or | bus_ex | p_or (MW) | q_or (MVAr) | v_or (kV) | a_or (A) | "
        "p_ex (MW) | q_ex (MVAr) | v_ex (kV) | a_ex (A) | rho | overflow | protect | maint | cooldown"
    )

    rows = []
    for line_id in range(getattr(obs, "n_line", 0)):
        try:
            or_local = int(obs.topo_vect[obs.line_or_pos_topo_vect[line_id]])
            ex_local = int(obs.topo_vect[obs.line_ex_pos_topo_vect[line_id]])
            or_sub = int(obs.line_or_to_subid[line_id])
            ex_sub = int(obs.line_ex_to_subid[line_id])
            or_bus = env.action_space.local_bus_to_global_int(or_local, or_sub)
            ex_bus = env.action_space.local_bus_to_global_int(ex_local, ex_sub)
            label = _line_label(env, obs, line_id)
            if obs.line_status[line_id]:
                por = _safe_get(lines_or.get("p", []), line_id, float("nan"))
                qor = _safe_get(lines_or.get("q", []), line_id, float("nan"))
                vor = _safe_get(lines_or.get("v", []), line_id, float("nan"))
                aor = _safe_get(lines_or.get("a", []), line_id, float("nan"))

                pex = _safe_get(lines_ex.get("p", []), line_id, float("nan"))
                qex = _safe_get(lines_ex.get("q", []), line_id, float("nan"))
                vex = _safe_get(lines_ex.get("v", []), line_id, float("nan"))
                aex = _safe_get(lines_ex.get("a", []), line_id, float("nan"))

                r = _safe_get(rho, line_id, float("nan"))
                ov = _safe_get(overflow, line_id, "NA")
                pr = _safe_get(protect, line_id, "NA")
                mt = _safe_get(maint_t, line_id, "-")
                md = _safe_get(maint_d, line_id, "-")
                cd = _safe_get(cooldown, line_id, 0)
                row = (
                    f"{label} | {or_bus} | {ex_bus} | "
                    f"{por:.1f} | {qor:.1f} | {vor:.1f} | {aor:.1f} | "
                    f"{pex:.1f} | {qex:.1f} | {vex:.1f} | {aex:.1f} | "
                    f"{r:.2f} | {ov} | {pr} | {mt}/{md} | {cd}"
                )
            else:
                ov = _safe_get(overflow, line_id, "NA")
                pr = _safe_get(protect, line_id, "NA")
                mt = _safe_get(maint_t, line_id, "-")
                md = _safe_get(maint_d, line_id, "-")
                cd = _safe_get(cooldown, line_id, 0)
                row = (
                    f"{label} | NA | NA | NA | NA | NA | NA | "
                    f"NA | NA | NA | NA | NA | {ov} | {pr} | {mt}/{md} | {cd}    # disabled {or_bus}-{ex_bus}"
                )
            rows.append(row)
        except Exception:
            rows.append(f"{line_id} | [error]")
    return caption + "\n" + "\n".join(bullets + [header] + rows)


def _format_generators(env, obs) -> str:
    """Table of generator outputs, limits and curtailment/redispatch info."""
    info = env.action_space.cls_to_dict()
    pmin = info.get("gen_pmin", [])
    pmax = info.get("gen_pmax", [])
    costs = info.get("gen_cost_per_MW", [])

    obs_dict = obs.to_dict()
    gens = obs_dict.get("gens", {})
    q = gens.get("q", [])
    v = gens.get("v", [])
    redispatch = obs_dict.get("redispatching", {})
    target_rd = redispatch.get("target_redispatch", [])
    actual_rd = redispatch.get("actual_dispatch", [])
    p_no_curt = obs_dict.get("gen_p_before_curtail", [])
    curtailed = obs_dict.get("curtailment", [])
    limit = obs_dict.get("curtailment_limit", [])
    eff_limit = obs_dict.get("curtailment_limit_effective", [])

    caption = "Generators (current output and limits):"
    bullets = [
        "- gen_id: generator identifier",
        "- p_MW/q_MVAr: active/reactive output",
        "- v_kV: terminal voltage",
        "- pmin/pmax: min and max active power",
        "- cost_per_MW: production cost coefficient",
        "- p_no_curt: power before curtailment",
        "- curtail: curtailment factor",
        "- limit/eff_lim: allowed curtailment and effective value",
        "- rd_target/rd_actual: redispatch setpoint vs executed",
        "- redispatchable generators have increments listed later",
        "- curtailable generators (renewables) have factors listed later",
    ]
    header = (
        "gen_id | p_MW | q_MVAr | v_kV | pmin (MW) | pmax (MW) | cost_per_MW | "
        "p_no_curt (MW) | curtail | limit | eff_lim | rd_target | rd_actual"
    )
    rows = []
    for gen_id in range(getattr(obs, "n_gen", 0)):
        try:
            p = float(obs.gen_p[gen_id])
            qv = _safe_get(q, gen_id, float("nan"))
            vv = _safe_get(v, gen_id, float("nan"))
            pmn = _safe_get(pmin, gen_id, "?")
            pmx = _safe_get(pmax, gen_id, "?")
            cost = _safe_get(costs, gen_id, "?")
            pnc = _safe_get(p_no_curt, gen_id, float("nan"))
            curt = _safe_get(curtailed, gen_id, float("nan"))
            lim = _safe_get(limit, gen_id, float("nan"))
            elim = _safe_get(eff_limit, gen_id, float("nan"))
            rdt = _safe_get(target_rd, gen_id, float("nan"))
            rda = _safe_get(actual_rd, gen_id, float("nan"))
            rows.append(
                f"gen_{gen_id} | {p:.1f} | {qv:.1f} | {vv:.1f} | {pmn} | {pmx} | {cost} | "
                f"{pnc:.1f} | {curt:.2f} | {lim:.2f} | {elim:.2f} | {rdt:.1f} | {rda:.1f}"
            )
        except Exception:
            rows.append(f"gen_{gen_id} | [error]")
    return caption + "\n" + "\n".join(bullets + [header] + rows)


def _format_loads(env, obs) -> str:
    if not hasattr(obs, "load_p"):
        return ""
    q = getattr(obs, "load_q", [])
    v = getattr(obs, "load_v", [])
    caption = "Loads (consumption per load):"
    bullets = [
        "- load_id: identifier of the load",
        "- p_MW/q_MVAr: active/reactive consumption",
        "- v_kV: voltage at the load bus",
    ]
    header = "load_id | p_MW | q_MVAr | v_kV"
    rows = []
    for load_id, p in enumerate(obs.load_p):
        try:
            qv = float(q[load_id]) if load_id < len(q) else float("nan")
            vv = float(v[load_id]) if load_id < len(v) else float("nan")
            rows.append(f"load_{load_id} | {float(p):.1f} | {qv:.1f} | {vv:.1f}")
        except Exception:
            rows.append(f"load_{load_id} | [error]")
    return caption + "\n" + "\n".join(bullets + [header] + rows)


def _format_storage(env, obs) -> str:
    if not hasattr(obs, "storage_power"):
        return ""
    if len(obs.storage_power) == 0:
        return ""
    info = env.action_space.cls_to_dict()
    max_prod = info.get("storage_max_p_prod", [])
    max_abs = info.get("storage_max_p_absorb", [])
    obs_dict = obs.to_dict()
    charge = obs_dict.get("storage_charge", [])
    target = obs_dict.get("storage_power_target", [])
    caption = "Storage (current power and state of charge):"
    bullets = [
        "- storage_id: storage unit identifier",
        "- p_MW: positive discharging, negative charging",
        "- soc: state of charge in MWh",
        "- target: target power setpoint (MW)",
        "- max_p_prod/max_p_absorb: discharge/charge limits (MW)",
    ]
    header = "storage_id | p_MW | soc | target | max_p_prod (MW) | max_p_absorb (MW)"
    rows = []
    for st_id in range(len(obs.storage_power)):
        try:
            p = float(obs.storage_power[st_id])
            mp = _safe_get(max_prod, st_id, "?")
            ma = _safe_get(max_abs, st_id, "?")
            soc = _safe_get(charge, st_id, float("nan"))
            tgt = _safe_get(target, st_id, float("nan"))
            rows.append(
                f"storage_{st_id} | {p:+.1f} | {soc:.1f} | {tgt:.1f} | {mp} | {ma}"
            )
        except Exception:
            rows.append(f"storage_{st_id} | [error]")
    return caption + "\n" + "\n".join(bullets + [header] + rows)


def _redispatch_values_per_gen(action_space) -> dict:
    """Compute discrete redispatch increments for each generator."""
    results = {}
    if hasattr(action_space, "gen_redispatchable"):
        num_down, num_up, max_ratio = (
            getattr(action_space.get_all_unitary_redispatch, "__defaults__")
        )
        for gen_id, dispatchable in enumerate(action_space.gen_redispatchable):
            if not dispatchable:
                continue
            vals = []
            ramp_up = float(action_space.gen_max_ramp_up[gen_id]) * max_ratio
            ramp_down = float(action_space.gen_max_ramp_down[gen_id]) * max_ratio
            if num_up > 0 and ramp_up > 0:
                step = ramp_up / num_up
                for i in range(1, num_up + 1):
                    vals.append(round(step * i, 1))
            if num_down > 0 and ramp_down > 0:
                step = ramp_down / num_down
                for i in range(1, num_down + 1):
                    vals.append(round(-step * i, 1))
            results[gen_id] = sorted(vals)
    return results


def _storage_values_per_storage(action_space) -> dict:
    """Compute discrete power values for each storage unit."""
    results = {}
    if getattr(action_space, "n_storage", 0) > 0:
        num_down, num_up = getattr(
            action_space.get_all_unitary_storage, "__defaults__"
        )
        for st_id in range(action_space.n_storage):
            vals = []
            max_up = float(action_space.storage_max_p_absorb[st_id])
            max_down = float(action_space.storage_max_p_prod[st_id])
            if num_up > 0 and max_up > 0:
                step = max_up / num_up
                for i in range(1, num_up + 1):
                    vals.append(round(step * i, 1))
            if num_down > 0 and max_down > 0:
                step = max_down / num_down
                for i in range(1, num_down + 1):
                    vals.append(round(-step * i, 1))
            results[st_id] = sorted(vals)
    return results


def _curtail_values_per_gen(action_space) -> dict:
    """Compute discrete curtailment factors for each renewable generator."""
    results = {}
    if hasattr(action_space, "gen_renewable"):
        num_bin, min_val = getattr(
            action_space.get_all_unitary_curtail, "__defaults__"
        )
        for gen_id, is_ren in enumerate(action_space.gen_renewable):
            if not is_ren:
                continue
            ramps = np.linspace(min_val, 1.0, num=num_bin)
            results[gen_id] = [round(float(r), 2) for r in ramps]
    return results


def _format_action_space(env) -> str:
    info = env.action_space.cls_to_dict()
    parts = [
        "=== ACTION CONSTRAINTS ===",
        "Redispatch adjusts the active power of dispatchable generators in MW steps",
        "Curtailment multiplies renewable generator output by a factor between 0 and 1",
        "Storage power is positive when discharging, negative when charging; state of charge is in MWh",
        f"Max busbars per substation: {info.get('n_busbar_per_sub', 2)}",
    ]

    rvals = _redispatch_values_per_gen(env.action_space)
    for gid, vals in rvals.items():
        parts.append(f"Redispatchable gen {gid} increments (MW): {vals}")

    cvals = _curtail_values_per_gen(env.action_space)
    for gid, vals in cvals.items():
        parts.append(f"Curtailable gen {gid} factors: {vals}")

    svals = _storage_values_per_storage(env.action_space)
    for sid, vals in svals.items():
        parts.append(f"Storage power values storage {sid} (MW): {vals}")

    return "\n".join(parts)


def build_prompt(env, obs) -> str:
    """Construct a textual prompt describing the grid state and action rules."""
    parts = []
    parts.append("OBJECTIVE: operate the grid safely while minimising cost.")
    parts.append(
        "Lines transport power between substations. Generators inject power, loads consume it, and storage can charge or discharge."
    )
    parts.append(
        "Redispatch changes generator output within ramp limits. Curtailment reduces renewable generation."
    )
    parts.append(
        f"Step {getattr(obs, 'current_step', '?')}/{getattr(obs, 'max_step', '?')}"
    )
    parts.append("=== CURRENT STATE ===")
    parts.append(_format_substations(env, obs))
    parts.append("")
    parts.append(_format_lines(env, obs))
    parts.append("")
    parts.append(_format_generators(env, obs))
    load_str = _format_loads(env, obs)
    if load_str:
        parts.append("")
        parts.append(load_str)
    stor = _format_storage(env, obs)
    if stor:
        parts.append("")
        parts.append(stor)

    parts.append("")
    parts.append(_format_action_space(env))

    parts.append("")
    parts.append("=== ACTION FORMATTING ===")
    parts.append(
        "Actions are JSON with EXACTLY ONE of these keys: set_line, set_bus, redispatch, curtail, storage. Each action affects only ONE element (one line, one substation, one generator, one storage unit). To do nothing, provide: {}"
    )
    parts.append(
        "Line names use format <bus-or>_<bus-ex>_<line-id>. The tables above follow the same ordering as bus vectors."
    )
    parts.append(
        "For set_bus: {\"set_bus\": {\"substations_id\": [(sub_id, [bus_assignments])]}}. Modify only ONE substation per action. Each element in bus_assignments corresponds to an element in the substation table, in the same order. The first element must be 1. Example: {\"set_bus\": {\"substations_id\": [(0, [1, 2, 1])]}}"
    )
    parts.append(
        "For set_line: {\"set_line\": {line_id: [bus_or, bus_ex]}} to reconnect, or {\"set_line\": {line_id: \"disconnect\"}} to disconnect. Examples: {\"set_line\": {0: [1, 2]}}, {\"set_line\": {3: \"disconnect\"}}"
    )
    parts.append(
        "For redispatch: {\"redispatch\": {\"gen_X\": MW_delta}} using values from the allowed increment list. Example: {\"redispatch\": {\"gen_1\": -5.0}}"
    )
    parts.append(
        "For curtail: {\"curtail\": {\"gen_X\": factor}} using values from the allowed curtailment factors. Example: {\"curtail\": {\"gen_2\": 0.8}}"
    )
    parts.append(
        "For storage: {\"storage\": {\"storage_X\": MW_value}} using values from the allowed power list. Example: {\"storage\": {\"storage_0\": 10.5}}"
    )
    return "\n".join(parts)

