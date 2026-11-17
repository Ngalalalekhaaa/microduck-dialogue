"""Independent geometric checks for the RGB-D marker measurement."""

import unittest

import numpy as np

from experiments.microduck_vision.vision import detect_target


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        self.depth = np.ones((100, 200), dtype=np.float64)
        self.position = np.array([0.05, 0.02, 0.3])
        self.rotation = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def detect(self, **overrides):
        arguments = {
            "rgb": self.rgb,
            "depth": self.depth,
            "fovy_deg": 90,
            "camera_position_body": self.position,
            "camera_rotation_body": self.rotation,
        }
        arguments.update(overrides)
        return detect_target(**arguments)

    def test_nonsquare_camera_uses_vertical_focal_length_and_camera_translation(self):
        self.rgb[45:55, 85:95, 0] = 255
        result = self.detect()
        self.assertTrue(result.found)
        self.assertAlmostEqual(result.forward_m, 1.05)
        self.assertAlmostEqual(result.left_m, 0.22)
        self.assertAlmostEqual(result.distance_m, np.hypot(1.05, 0.22))
        self.assertEqual(result.pixel_uv, (89.5, 49.5))
        self.assertEqual(result.pixels, 100)

    def test_image_right_maps_to_robot_right(self):
        self.rgb[45:55, 105:115, 0] = 255
        result = self.detect()
        self.assertAlmostEqual(result.left_m, -0.18)

    def test_centered_even_sized_image_has_no_half_pixel_bias(self):
        self.rgb[45:55, 95:105, 0] = 255
        result = self.detect()
        self.assertAlmostEqual(result.left_m, 0.02)
        self.assertEqual(result.pixel_uv, (99.5, 49.5))

    def test_pitched_camera_rotates_depth_into_body_frame(self):
        self.rgb[45:55, 95:105, 0] = 255
        angle = np.pi / 6
        pitch = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        result = self.detect(camera_rotation_body=pitch @ self.rotation)
        self.assertAlmostEqual(result.forward_m, np.cos(angle) + 0.05)
        self.assertAlmostEqual(result.left_m, 0.02)

    def test_no_red_is_missing(self):
        result = self.detect()
        self.assertFalse(result.found)
        self.assertIsNone(result.forward_m)
        self.assertIsNone(result.left_m)
        self.assertIsNone(result.distance_m)
        self.assertIsNone(result.pixel_uv)

    def test_largest_connected_component_ignores_separate_red_noise(self):
        self.rgb[45:55, 95:105, 0] = 255
        self.rgb[0:3, 0:3, 0] = 255
        result = self.detect()
        self.assertEqual(result.pixels, 100)
        self.assertAlmostEqual(result.left_m, 0.02)

    def test_invalid_depth_is_excluded_and_outlier_does_not_move_median(self):
        self.rgb[45:55, 95:105, 0] = 255
        self.depth[45, 95:100] = [np.nan, np.inf, -np.inf, 0, -1]
        self.depth[46, 95] = 100
        result = self.detect()
        self.assertTrue(result.found)
        self.assertEqual(result.pixels, 95)
        self.assertAlmostEqual(result.forward_m, 1.05)
        self.assertTrue(np.isfinite(result.distance_m))

    def test_red_with_no_usable_depth_is_missing(self):
        self.rgb[45:55, 95:105, 0] = 255
        self.depth[:] = np.nan
        result = self.detect()
        self.assertFalse(result.found)
        self.assertEqual(result.reason, "invalid_target_depth")
        self.assertEqual(result.pixels, 0)

    def test_tiny_component_does_not_trigger_motion(self):
        self.rgb[50, 100, 0] = 255
        self.assertFalse(self.detect().found)

    def test_invalid_geometry_raises_actionable_error(self):
        cases = [
            {"rgb": self.rgb[..., 0]},
            {"rgb": self.rgb.astype(float)},
            {"depth": self.depth[:10]},
            {"fovy_deg": 0},
            {"fovy_deg": 180},
            {"fovy_deg": float("nan")},
            {"camera_position_body": [0, 0, np.inf]},
            {"camera_rotation_body": np.ones((3, 3))},
            {"camera_rotation_body": -np.eye(3)},
            {"min_pixels": 0},
            {"min_pixels": 1.5},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.detect(**case)


if __name__ == "__main__":
    unittest.main()
