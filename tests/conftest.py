import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

# Provide a stub for the treesearch backend to avoid heavy dependencies
import types

class DummySpec:
    def __init__(self, *args, **kwargs):
        pass

stub_backend = types.SimpleNamespace(
    query=lambda **kwargs: {"selected_id": "", "reasoning": ""},
    FunctionSpec=DummySpec,
)
sys.modules.setdefault("ai_scientist.treesearch.backend", stub_backend)

# Stub out heavy LLM dependencies
token_stub = types.SimpleNamespace(track_token_usage=lambda f: f)
sys.modules.setdefault("ai_scientist.utils.token_tracker", token_stub)

class DummyClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))])

def create_client(model):
    return DummyClient(), model

llm_stub = types.SimpleNamespace(create_client=create_client)
sys.modules.setdefault("ai_scientist.llm", llm_stub)

sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))

sys.modules.setdefault("ai_scientist.perform_llm_review", types.SimpleNamespace(perform_review=lambda *a, **k: {}))
sys.modules.setdefault(
    "ai_scientist.perform_vlm_review",
    types.SimpleNamespace(
        perform_imgs_cap_ref_review=lambda *a, **k: {},
        detect_duplicate_figures=lambda *a, **k: {},
    ),
)
sys.modules.setdefault(
    "ai_scientist.perform_icbinb_writeup",
    types.SimpleNamespace(
        gather_citations=lambda *a, **k: None,
        compile_latex=lambda cwd, pdf: Path(pdf).write_text(""),
        get_reflection_page_info=lambda pdf, limit: "",
    ),
)
sys.modules.setdefault("backoff", types.ModuleType("backoff"))
sys.modules.setdefault("openai", types.ModuleType("openai"))
vlm_stub = types.SimpleNamespace(create_client=create_client)
sys.modules.setdefault("ai_scientist.vlm", vlm_stub)
