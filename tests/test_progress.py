"""Tests for visage.progress — progress display with quiet/plain-text modes."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from visage.progress import ProgressDisplay


class TestQuietMode:
    def test_update_suppressed(self, capsys):
        pd = ProgressDisplay(quiet=True)
        pd.update("1/5 Test", 5, 10)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_finish_phase_suppressed(self, capsys):
        pd = ProgressDisplay(quiet=True)
        pd.finish_phase("1/5 Test", "Done")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_error_suppressed(self, capsys):
        pd = ProgressDisplay(quiet=True)
        pd.error("bad thing")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_finish_suppressed(self, capsys):
        pd = ProgressDisplay(quiet=True)
        pd.finish("summary")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestPlainTextFallback:
    def test_update_output(self, capsys):
        # Force plain text by making _ensure_rich return False
        with patch("visage.progress._RICH_AVAILABLE", False):
            pd = ProgressDisplay(quiet=False)
            pd.update("1/5 Test", 5, 10)
            captured = capsys.readouterr()
            assert "1/5 Test" in captured.err
            assert "5/10" in captured.err

    def test_finish_phase_output(self, capsys):
        with patch("visage.progress._RICH_AVAILABLE", False):
            pd = ProgressDisplay(quiet=False)
            pd.finish_phase("1/5 Scan", "Found 42 images")
            captured = capsys.readouterr()
            assert "1/5 Scan" in captured.err
            assert "Found 42 images" in captured.err

    def test_error_output(self, capsys):
        with patch("visage.progress._RICH_AVAILABLE", False):
            pd = ProgressDisplay(quiet=False)
            pd.error("something went wrong")
            captured = capsys.readouterr()
            assert "ERROR" in captured.err
            assert "something went wrong" in captured.err

    def test_finish_output(self, capsys):
        with patch("visage.progress._RICH_AVAILABLE", False):
            pd = ProgressDisplay(quiet=False)
            pd.finish("All done!")
            captured = capsys.readouterr()
            assert "All done!" in captured.err

    def test_print_plan_output(self, capsys):
        with patch("visage.progress._RICH_AVAILABLE", False):
            pd = ProgressDisplay(quiet=False)
            pd.print_plan("person_00: 5 photos")
            captured = capsys.readouterr()
            assert "person_00" in captured.err

    def test_update_with_extra(self, capsys):
        with patch("visage.progress._RICH_AVAILABLE", False):
            pd = ProgressDisplay(quiet=False)
            pd.update("2/5 Detect", 3, 10, extra="processing photo.jpg")
            captured = capsys.readouterr()
            assert "processing photo.jpg" in captured.err
