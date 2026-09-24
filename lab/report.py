"""Collects the markdown written to results.md, and prints section banners."""
from pathlib import Path


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def section(self, title: str) -> None:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        self.add("", f"## {title}", "")

    def table(self, header: list[str], rows: list[list[object]]) -> None:
        self.add("| " + " | ".join(header) + " |")
        self.add("|" + "---|" * len(header))
        for row in rows:
            self.add("| " + " | ".join(str(cell) for cell in row) + " |")
        self.add("")

    def paragraph(self, *sentences: str) -> None:
        self.add(" ".join(sentences), "")

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def code_list(names: list[str]) -> str:
    """Format names as `a`, `b` and `c` for the written answers."""
    quoted = [f"`{name}`" for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]
