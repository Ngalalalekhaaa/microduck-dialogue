"""Conservative interior of the measured simulated foot-contact envelope.

Evidence: outputs/kick_reach_20261115/reach.json. This is not a hardware calibration.
The legacy trained prompt retains its original region; v2/efficient teachers use this.
"""

import math


def reachable(point, foot):
    if foot not in ("left", "right") or len(point) != 3 or not all(math.isfinite(v) for v in point):
        return False
    x, y, z = point
    lateral = y if foot == "left" else -y
    # Center-line hits passed reset-pose probes but miss after some walking
    # stances (q1r physical sweep). Require a lateral margin for either foot.
    far = 0.105
    return 0.075 <= x <= far and 0.015 <= lateral <= 0.055 and -0.11 <= z <= -0.04


def choose_foot(point):
    order = ("left", "right") if point[1] > 0.005 else ("right", "left")
    return next((foot for foot in order if reachable(point, foot)), None)
