"""Three kick balls, a free grasp toy, a delivery mat, and an articulated beak.

Beak pad dimensions adapted from llama/microduck-beak-throw (Apache-2.0),
commit 410ed2249d62cd7aa4c642cd2c4e9b56f9a0cbfd. See ACTIONS.md for attribution.
This is a simulation attachment, not a claim about the stock fixed-jaw MJCF.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .scene import build_scene

TOY_RADIUS = 0.012
MOUTH_OPEN = np.deg2rad(30)
MOUTH_HOLD = np.deg2rad(20)
POCKET_LOCAL = np.array([-0.0155, 0, -0.0687])


def build_action_scene(path, *, toy_xy=(0.3, 0.0), support_height=0.0, block_xy=(0.45, 0.32)):
    build_scene(path)
    root = ET.fromstring(path.read_text())
    head = next(b for b in root.iter("body") if b.get("name") == "jaw_soft")
    pivot = np.array([0.003, 0, -0.018])
    lower = ET.SubElement(head, "body", name="beak_lower", pos="0.003 0 -0.018")
    ET.SubElement(
        lower,
        "joint",
        name="passive_mouth",
        type="hinge",
        axis="0 1 0",
        range="-0.0872665 0.523599",
        damping="0.02",
        armature="0.00001",
        frictionloss="0",
    )
    ET.SubElement(lower, "inertial", pos="-0.018 0 -0.027", mass="0.015", diaginertia="4e-6 6e-6 3e-6")
    inertial = head.find("inertial")
    inertial.set("mass", str(float(inertial.get("mass")) - 0.015))
    for geom in list(head.findall("geom")):
        mesh = geom.get("mesh")
        if mesh in ("jaw", "jaw_soft"):
            head.remove(geom)
            if geom.get("class") == "visual":
                pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ") - pivot
                geom.set("pos", " ".join(map(str, pos)))
                lower.append(geom)
        elif mesh == "bottom_head_shell" and geom.get("class") == "collision":
            head.remove(geom)
        elif geom.get("class") == "collision":
            # Convex head hulls have no mouth cavity. Keep floor/body contacts,
            # but let the small toy contact the explicit palate geometry.
            geom.set("contype", "4")
            geom.set("conaffinity", "1")

    def pad(body, name, pos, size, *, capsule=False):
        attributes = dict(
            name=name,
            type="capsule" if capsule else "box",
            size=size,
            contype="2",
            conaffinity="3",
            condim="6",
            priority="1",
            friction="1.0 0.01 0.001",
            solref="0.012 1",
            group="2",
            rgba="0.95 0.6 0.04 1",
            mass="0",
        )
        attributes["fromto" if capsule else "pos"] = pos
        ET.SubElement(body, "geom", **attributes)

    pad(head, "grip_upper_pad", "-0.002 0 -0.0687", "0.0015 0.017 0.018")
    pad(lower, "grip_lower_pad", "-0.013 0 -0.054", "0.0015 0.017 0.018")
    pad(head, "grip_upper_lip", "-0.007 -0.017 -0.080 -0.007 0.017 -0.080", "0.002", capsule=True)
    pad(lower, "grip_lower_lip", "-0.007 -0.017 -0.066 -0.007 0.017 -0.066", "0.002", capsule=True)
    pad(head, "grip_rear_stop", "-0.0155 0 -0.0545", "0.013 0.017 0.0015")
    for side, y in (("left", 0.016), ("right", -0.016)):
        pad(head, f"grip_{side}_cheek", f"-0.0155 {y} -0.068", "0.013 0.002 0.015")
    ET.SubElement(
        root.find("actuator"),
        "position",
        name="mouth_servo",
        joint="passive_mouth",
        kp="0.45",
        kv="0.008",
        ctrllimited="true",
        ctrlrange="-0.0872665 0.523599",
        forcelimited="true",
        forcerange="-0.15 0.15",
    )
    world = root.find("worldbody")
    toy = ET.SubElement(world, "body", name="grasp_toy", pos=f"{toy_xy[0]} {toy_xy[1]} {TOY_RADIUS + support_height}")
    ET.SubElement(toy, "freejoint", name="grasp_toy_free")
    ET.SubElement(
        toy,
        "geom",
        name="grasp_toy_geom",
        type="sphere",
        size="0.012",
        mass="0.003",
        rgba="0.05 0.9 0.25 1",
        friction="0.8 0.01 0.001",
        condim="6",
        contype="2",
        conaffinity="3",
        solref="0.012 1",
    )
    if support_height:
        ET.SubElement(
            world,
            "geom",
            name="toy_support",
            type="box",
            pos=f"{toy_xy[0]} {toy_xy[1]} {support_height / 2}",
            size=f"0.020 0.020 {support_height / 2}",
            rgba="0.45 0.6 0.65 1",
        )
    ET.SubElement(
        world,
        "geom",
        name="delivery_mat",
        type="box",
        pos="0.32 0 0.001",
        size="0.10 0.10 0.001",
        rgba="0.2 0.75 0.7 0.7",
        contype="0",
        conaffinity="0",
    )
    box = ET.SubElement(world, "body", name="push_block", pos=f"{block_xy[0]} {block_xy[1]} 0.025")
    ET.SubElement(box, "freejoint", name="push_block_free")
    ET.SubElement(
        box,
        "geom",
        name="push_block_geom",
        type="box",
        size="0.025 0.035 0.025",
        mass="0.02",
        rgba="0.75 0.25 0.8 1",
        friction="0.45 0.005 0.0001",
    )
    path.write_text(ET.tostring(root, encoding="unicode"))
    mujoco.MjModel.from_xml_path(str(path))
