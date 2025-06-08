from pathlib import Path
import types
from ai_scientist.paper_improver import latex_editor


def test_propose_edit_extract(monkeypatch, tmp_path):
    tex = tmp_path / "t.tex"
    tex.write_text("Original")

    class DummyCompletions:
        @staticmethod
        def create(**kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="```latex\nEDITED\n```"))]
            )

    class DummyClient:
        chat = types.SimpleNamespace(completions=DummyCompletions)

    monkeypatch.setattr(latex_editor, "create_client", lambda model: (DummyClient(), model))

    result = latex_editor.propose_edit(tex, "idea", model="dummy")
    assert result == "EDITED"
