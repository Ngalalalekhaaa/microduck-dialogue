"""Independent ray/sphere ground truth for the RGB-D ball detector."""

import numpy as np
import pytest

from experiments.microduck_tasks.perception import detect_balls

COLOURS = {"red": [230, 12, 12], "yellow": [230, 220, 12], "blue": [12, 35, 230]}
ROTATION = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
POSITION = np.array([0.12, 0.02, 0.16])


def render(spheres, *, shape=(240, 320), fovy=85):
    """Generate exact optical depth via quadratic ray/sphere intersections."""
    height, width = shape
    v, u = np.indices(shape)
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    rays = np.stack(((u - (width - 1) / 2) / focal, -(v - (height - 1) / 2) / focal, -np.ones(shape)), -1)
    a = np.sum(rays**2, axis=-1)
    rgb = np.full((*shape, 3), 100, dtype=np.uint8)
    depth = np.full(shape, 4.0)
    for colour, centre, radius in spheres:
        centre = np.asarray(centre)
        b = np.sum(rays * centre, axis=-1)
        discriminant = b**2 - a * (np.dot(centre, centre) - radius**2)
        t = (b - np.sqrt(np.maximum(0, discriminant))) / a
        hit = (discriminant >= 0) & (t > 0) & (t < depth)
        rgb[hit] = COLOURS[colour]
        depth[hit] = t[hit]
    return rgb, depth


def detect(rgb, depth, **overrides):
    arguments = dict(fovy_deg=85, camera_position_body=POSITION, camera_rotation_body=ROTATION)
    arguments.update(overrides)
    return detect_balls(rgb, depth, **arguments)


def test_three_colours_recover_centres_with_translation_and_axes():
    spheres = [
        ("red", [-0.12, 0.02, -0.45], 0.035),
        ("yellow", [0, 0, -0.6], 0.035),
        ("blue", [0.16, -0.02, -0.7], 0.035),
    ]
    rgb, depth = render(spheres)
    found = {entry["color"]: entry for entry in detect(rgb, depth)}
    assert set(found) == {"red", "yellow", "blue"}
    for colour, centre, _ in spheres:
        entry = found[colour]
        np.testing.assert_allclose(entry["center_body_m"], ROTATION @ centre + POSITION, atol=1e-7)
        assert entry["pixels"] >= 10
        assert 0 < entry["confidence"] <= 1
        assert entry["fit_residual_m"] < 1e-7
        assert entry["camera"] == "front"


def test_separate_balls_of_same_colour_are_preserved():
    rgb, depth = render([("red", [-0.12, 0, -0.4], 0.035), ("red", [0.12, 0, -0.5], 0.035)])
    assert len(detect(rgb, depth)) == 2


def test_partial_sphere_with_noise_bad_depth_and_outliers_recovers_centre():
    centre = np.array([0.02, 0.02, -0.45])
    rgb, depth = render([("yellow", centre, 0.035)])
    rgb[:, 175:] = 100
    rows, columns = np.where(rgb[..., 0] == 230)
    rng = np.random.default_rng(240)
    depth[rows, columns] += rng.normal(0, 0.0004, len(rows))
    depth[rows[::20], columns[::20]] = np.nan
    depth[rows[1::20], columns[1::20]] = np.inf
    depth[rows[2::20], columns[2::20]] += 2
    found = detect(rgb, depth, camera_name="near")
    assert len(found) == 1
    np.testing.assert_allclose(found[0]["center_body_m"], ROTATION @ centre + POSITION, atol=0.002)
    assert found[0]["camera"] == "near"


@pytest.mark.parametrize("noise", [0, 0.0003])
def test_coloured_plane_is_not_a_ball(noise):
    rgb = np.full((240, 320, 3), 100, dtype=np.uint8)
    depth = np.full((240, 320), 0.5)
    rgb[110:130, 145:175] = COLOURS["blue"]
    depth += np.random.default_rng(4).normal(0, noise, depth.shape)
    assert detect(rgb, depth) == []


def test_wrong_radius_is_rejected():
    rgb, depth = render([("red", [0, 0, -0.5], 0.08)])
    assert detect(rgb, depth) == []


def test_orange_robot_colour_is_not_a_red_or_yellow_ball():
    rgb, depth = render([("red", [0, 0, -0.5], 0.035)])
    rgb[rgb[..., 0] == 230] = [240, 120, 10]
    assert detect(rgb, depth) == []


def test_muted_orange_foot_highlight_is_not_yellow():
    rgb, depth = render([("yellow", [0, 0, -0.3], 0.035)])
    rgb[rgb[..., 0] == 230] = [100, 75, 25]
    assert detect(rgb, depth) == []


def test_thin_curved_strip_cannot_constrain_a_sphere_centre():
    rgb, depth = render([("yellow", [0, 0, -0.3], 0.035)])
    # Even exact spherical curvature is insufficient when only a narrow strip
    # is observed: this covers the small bent highlight on the robot's foot.
    rgb[:119] = 100
    rgb[121:] = 100
    assert np.count_nonzero(rgb[..., 0] == 230) >= 10
    assert detect(rgb, depth) == []


def test_empty_or_tiny_regions_do_not_produce_balls():
    rgb, depth = render([])
    assert detect(rgb, depth) == []
    rgb[10:12, 10:12] = COLOURS["red"]
    assert detect(rgb, depth) == []


def test_far_sphere_still_detected_and_correctly_localised():
    centre = [0.2, 0, -1.6]
    rgb, depth = render([("blue", centre, 0.035)])
    found = detect(rgb, depth)
    assert len(found) == 1
    np.testing.assert_allclose(found[0]["center_body_m"], ROTATION @ centre + POSITION, atol=1e-7)


def test_all_invalid_depth_is_missing():
    rgb, depth = render([("red", [0, 0, -0.5], 0.035)])
    depth[:] = np.nan
    assert detect(rgb, depth) == []


@pytest.mark.parametrize(
    "override",
    [
        {"rgb": np.zeros((20, 30), np.uint8)},
        {"rgb": np.zeros((20, 30, 3), float)},
        {"depth": np.ones((20, 30))},
        {"fovy_deg": 0},
        {"fovy_deg": 180},
        {"fovy_deg": float("nan")},
        {"camera_position_body": [0, 0, float("inf")]},
        {"camera_rotation_body": np.ones((3, 3))},
        {"camera_rotation_body": -np.eye(3)},
        {"radius": 0},
        {"radius": float("nan")},
        {"camera_name": ""},
    ],
)
def test_invalid_inputs_raise_actionable_error(override):
    rgb, depth = render([])
    arguments = dict(rgb=rgb, depth=depth, fovy_deg=85, camera_position_body=POSITION, camera_rotation_body=ROTATION)
    arguments.update(override)
    with pytest.raises(ValueError):
        detect_balls(**arguments)
