"""Library data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Library:
    """Represents an independent photo library."""

    library_id: str
    name: str
    input_dir: str
    created_at: float = 0.0
    last_opened_at: float = 0.0
    photo_count: int = 0
    cluster_count: int = 0
    face_count: int = 0
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "library_id": self.library_id,
            "name": self.name,
            "input_dir": self.input_dir,
            "created_at": self.created_at,
            "last_opened_at": self.last_opened_at,
            "photo_count": self.photo_count,
            "cluster_count": self.cluster_count,
            "face_count": self.face_count,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Library:
        return cls(
            library_id=data["library_id"],
            name=data["name"],
            input_dir=data["input_dir"],
            created_at=data.get("created_at", 0.0),
            last_opened_at=data.get("last_opened_at", 0.0),
            photo_count=data.get("photo_count", 0),
            cluster_count=data.get("cluster_count", 0),
            face_count=data.get("face_count", 0),
            settings=data.get("settings", {}),
        )
