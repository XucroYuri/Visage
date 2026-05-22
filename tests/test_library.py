"""Tests for multi-library management."""

from __future__ import annotations

from visage.library.manager import LibraryManager
from visage.library.model import Library


class TestLibraryModel:
    """Test Library data model."""

    def test_to_dict(self):
        lib = Library(
            library_id="abc123",
            name="Test Library",
            input_dir="/tmp/photos",
            photo_count=100,
        )
        d = lib.to_dict()
        assert d["library_id"] == "abc123"
        assert d["name"] == "Test Library"
        assert d["photo_count"] == 100
        assert d["settings"] == {}

    def test_from_dict(self):
        data = {
            "library_id": "xyz",
            "name": "My Library",
            "input_dir": "/photos",
            "photo_count": 50,
            "cluster_count": 5,
        }
        lib = Library.from_dict(data)
        assert lib.library_id == "xyz"
        assert lib.photo_count == 50
        assert lib.cluster_count == 5

    def test_roundtrip(self):
        lib = Library(library_id="r1", name="R", input_dir="/r")
        restored = Library.from_dict(lib.to_dict())
        assert restored.library_id == lib.library_id
        assert restored.name == lib.name


class TestLibraryManager:
    """Test library CRUD operations."""

    def test_create_library(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        lib = mgr.create_library("Vacation 2024", "/tmp/vacation")
        assert lib.library_id
        assert lib.name == "Vacation 2024"
        assert lib.input_dir == "/tmp/vacation"
        assert lib.created_at > 0
        mgr.close()

    def test_get_library(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        created = mgr.create_library("Test", "/tmp/test")
        fetched = mgr.get_library(created.library_id)
        assert fetched is not None
        assert fetched.name == "Test"
        mgr.close()

    def test_get_nonexistent(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        assert mgr.get_library("nonexistent") is None
        mgr.close()

    def test_list_libraries(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        mgr.create_library("A", "/a")
        mgr.create_library("B", "/b")
        libs = mgr.list_libraries()
        assert len(libs) == 2
        mgr.close()

    def test_list_sorted_by_recent(self, tmp_path):
        import time
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        lib_a = mgr.create_library("First", "/a")
        time.sleep(0.01)
        mgr.create_library("Second", "/b")
        # Update first to make it more recent
        mgr.update_library(lib_a.library_id, name="First Updated")
        libs = mgr.list_libraries()
        assert libs[0].library_id == lib_a.library_id  # Most recently opened
        mgr.close()

    def test_update_library(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        lib = mgr.create_library("Original", "/original")
        updated = mgr.update_library(
            lib.library_id,
            name="Renamed",
            photo_count=42,
            cluster_count=5,
        )
        assert updated is not None
        assert updated.name == "Renamed"
        assert updated.photo_count == 42
        assert updated.cluster_count == 5
        mgr.close()

    def test_update_nonexistent(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        result = mgr.update_library("nope", name="X")
        assert result is None
        mgr.close()

    def test_update_settings(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        lib = mgr.create_library("Test", "/test")
        mgr.update_library(lib.library_id, settings={"dark_mode": True})
        fetched = mgr.get_library(lib.library_id)
        assert fetched.settings == {"dark_mode": True}
        mgr.close()

    def test_delete_library(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        lib = mgr.create_library("Delete Me", "/delete")
        assert mgr.delete_library(lib.library_id) is True
        assert mgr.get_library(lib.library_id) is None
        mgr.close()

    def test_delete_nonexistent(self, tmp_path):
        mgr = LibraryManager(base_path=str(tmp_path / "visage"))
        assert mgr.delete_library("nope") is False
        mgr.close()

    def test_persistence(self, tmp_path):
        base = str(tmp_path / "visage")
        mgr1 = LibraryManager(base_path=base)
        mgr1.create_library("Persist", "/persist")
        mgr1.close()

        # Reopen
        mgr2 = LibraryManager(base_path=base)
        libs = mgr2.list_libraries()
        assert len(libs) == 1
        assert libs[0].name == "Persist"
        mgr2.close()
