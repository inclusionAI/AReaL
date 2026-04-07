"""Tests for maze_verifier module."""

import numpy as np
import pytest
from PIL import Image

from geo_edit.evaluation.maze_verifier import (
    extract_maze_answer,
    parse_gt_path,
    get_maze_bounds,
    is_valid_step,
    wall_judge,
    maze_judge,
    _bresenham_line,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_open_maze(size=512) -> np.ndarray:
    """Create a simple open maze: white interior with a thin black border."""
    gray = np.full((size, size), 200, dtype=np.uint8)  # gray interior (passable)
    gray[0, :] = 0  # top wall
    gray[-1, :] = 0  # bottom wall
    gray[:, 0] = 0  # left wall
    gray[:, -1] = 0  # right wall
    return gray


def _make_maze_with_wall(size=512) -> np.ndarray:
    """Create a maze with a vertical wall in the middle (column 256)."""
    gray = _make_open_maze(size)
    gray[1:-1, 256] = 0  # vertical wall from row 1 to row 510
    return gray


def _save_temp_maze(tmp_path, gray_array: np.ndarray, name="maze.png") -> str:
    """Save a grayscale array as PNG and return the path."""
    path = str(tmp_path / name)
    Image.fromarray(gray_array).save(path)
    return path


# ---------------------------------------------------------------------------
# extract_maze_answer
# ---------------------------------------------------------------------------


class TestExtractMazeAnswer:
    def test_valid_single_sequence(self):
        text = "<point>500 500</point><point>600 600</point>"
        result = extract_maze_answer(text)
        assert len(result) == 2
        assert result[0] == (500, 500)
        assert result[1] == (600, 600)

    def test_takes_last_sequence(self):
        text = "<point>100 100</point> some text <point>500 500</point><point>600 600</point>"
        result = extract_maze_answer(text)
        # Should take the last consecutive sequence
        assert len(result) == 2

    def test_empty_string(self):
        assert extract_maze_answer("") == []

    def test_no_point_tags(self):
        assert extract_maze_answer("The answer is 42") == []

    def test_malformed_tags(self):
        assert extract_maze_answer("<point>abc def</point>") == []

    def test_single_point(self):
        result = extract_maze_answer("<point>0 0</point>")
        assert len(result) == 1
        assert result[0] == (0, 0)

    def test_coordinates_at_boundary(self):
        result = extract_maze_answer("<point>1000 1000</point>")
        assert len(result) == 1
        assert result[0] == (1000, 1000)

    def test_zero_coordinates(self):
        result = extract_maze_answer("<point>0 0</point>")
        assert result[0] == (0, 0)


# ---------------------------------------------------------------------------
# parse_gt_path
# ---------------------------------------------------------------------------


class TestParseGtPath:
    def test_basic(self):
        result = parse_gt_path("100 200 300 400")
        assert result == [(100, 200), (300, 400)]

    def test_single_point(self):
        result = parse_gt_path("50 60")
        assert result == [(50, 60)]

    def test_odd_numbers_truncated(self):
        # Odd number of ints: last one dropped
        result = parse_gt_path("10 20 30")
        assert result == [(10, 20)]


# ---------------------------------------------------------------------------
# get_maze_bounds
# ---------------------------------------------------------------------------


class TestGetMazeBounds:
    def test_detects_border(self):
        gray = _make_open_maze(100)
        x_min, x_max, y_min, y_max = get_maze_bounds(gray)
        assert x_min == 0
        assert x_max == 99
        assert y_min == 0
        assert y_max == 99

    def test_white_image_raises(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        with pytest.raises(ValueError, match="No maze region"):
            get_maze_bounds(gray)

    def test_small_region(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        gray[30:50, 40:60] = 100  # small dark patch
        x_min, x_max, y_min, y_max = get_maze_bounds(gray)
        assert x_min == 40
        assert x_max == 59
        assert y_min == 30
        assert y_max == 49


# ---------------------------------------------------------------------------
# bresenham_line
# ---------------------------------------------------------------------------


class TestBresenhamLine:
    def test_horizontal(self):
        rr, cc = _bresenham_line(5, 0, 5, 10)
        assert len(rr) == 11
        assert all(r == 5 for r in rr)
        assert list(cc) == list(range(11))

    def test_vertical(self):
        rr, cc = _bresenham_line(0, 5, 10, 5)
        assert len(rr) == 11
        assert all(c == 5 for c in cc)
        assert list(rr) == list(range(11))

    def test_single_point(self):
        rr, cc = _bresenham_line(3, 7, 3, 7)
        assert len(rr) == 1
        assert rr[0] == 3 and cc[0] == 7


# ---------------------------------------------------------------------------
# is_valid_step
# ---------------------------------------------------------------------------


class TestIsValidStep:
    def test_open_path(self):
        gray = _make_open_maze(100)
        valid, reason = is_valid_step(gray, (10, 10), (50, 50))
        assert valid is True

    def test_wall_collision(self):
        gray = _make_maze_with_wall(512)
        # Try to cross the vertical wall at column 256
        valid, reason = is_valid_step(gray, (200, 256), (300, 256))
        assert valid is False
        assert "wall" in reason.lower() or "Wall" in reason

    def test_endpoint_on_border_wall(self):
        gray = _make_open_maze(100)
        # Point on the border wall (row 0)
        valid, reason = is_valid_step(gray, (50, 0), (50, 50))
        assert valid is False

    def test_out_of_bounds(self):
        gray = _make_open_maze(100)
        valid, reason = is_valid_step(gray, (50, 50), (200, 200))
        assert valid is False


# ---------------------------------------------------------------------------
# wall_judge
# ---------------------------------------------------------------------------


class TestWallJudge:
    def test_empty_prediction(self, tmp_path):
        gray = _make_open_maze(512)
        path = _save_temp_maze(tmp_path, gray)
        score, _ = wall_judge("no points here", path, "100 100 200 200")
        assert score == 0

    def test_single_point_returns_zero(self, tmp_path):
        gray = _make_open_maze(512)
        path = _save_temp_maze(tmp_path, gray)
        score, _ = wall_judge("<point>100 100</point>", path, "50 50 200 200")
        assert score == 0

    def test_perfect_path(self, tmp_path):
        """Pred in 0-1000 space that maps to GT pixel coords should score > 0.

        GT pixel (100,100)->(400,100) on 512px ≈ 0-1000 coords (195,195)->(781,195).
        """
        gray = _make_open_maze(512)
        path = _save_temp_maze(tmp_path, gray)
        gt = "100 100 400 100"
        pred = "<point>195 195</point><point>781 195</point>"
        score, _ = wall_judge(pred, path, gt)
        assert score >= 0.5

    def test_start_too_far_returns_zero(self, tmp_path):
        gray = _make_open_maze(512)
        path = _save_temp_maze(tmp_path, gray)
        # GT starts at (10, 10), prediction starts far away
        gt = "10 10 200 200"
        pred = "<point>900 900</point><point>950 950</point>"
        score, _ = wall_judge(pred, path, gt)
        assert score == 0


# ---------------------------------------------------------------------------
# maze_judge (TrajectoryJudge-compatible interface)
# ---------------------------------------------------------------------------


class TestMazeJudge:
    def test_returns_tuple(self, tmp_path):
        gray = _make_open_maze(512)
        path = _save_temp_maze(tmp_path, gray)
        result = maze_judge("question", "100 100 200 200", "no points", path)
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_correct, reason = result
        assert isinstance(is_correct, bool)
        assert isinstance(reason, str)

    def test_no_path_returns_false(self, tmp_path):
        gray = _make_open_maze(512)
        path = _save_temp_maze(tmp_path, gray)
        is_correct, reason = maze_judge("q", "100 100 200 200", "no answer", path)
        assert is_correct is False
        assert "no_path" in reason or "insufficient" in reason

    def test_missing_image_returns_false(self):
        is_correct, reason = maze_judge(
            "q", "100 100", "<point>100 100</point>", "/nonexistent.png"
        )
        assert is_correct is False
        assert "error" in reason
