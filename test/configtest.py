# test/conftest.py
import pytest
import os
import pathlib
import sys
import shutil

# --- PATH SETUP (Crucial for imports to work) ---
ROOT = pathlib.Path(__file__).resolve().parents[1]  # Adjust based on where conftest.py lives
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- ENVIRONMENT VARIABLES ---
os.environ.setdefault("PYTHONPATH", str(ROOT / "multi_doc_chat"))
os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("LLM_PROVIDER", "google")

# --- MOCK CLASSES ---
class _StubEmbeddings:
    def embed_query(self, text: str): return [0.0, 0.1, 0.2]
    def embed_documents(self, texts): return [[0.0, 0.1, 0.2] for _ in texts]
    def __call__(self, text: str): return [0.0, 0.1, 0.2]

class _StubLLM:
    def invoke(self, input): return "stubbed answer"

# --- SHARED FIXTURES ---

@pytest.fixture
def tmp_dirs(tmp_path: pathlib.Path):
    data_dir = tmp_path / "data"
    faiss_dir = tmp_path / "faiss_index"
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    
    cwd = pathlib.Path.cwd()
    try:
        # Switch to temp dir so code writing files writes here
        os.chdir(tmp_path)
        yield {"data": data_dir, "faiss": faiss_dir}
    finally:
        os.chdir(cwd)

@pytest.fixture
def stub_model_loader(monkeypatch):
    # Depending on your file structure, you might need to try/except imports
    # or ensure sys.path is correct before these imports run.
    import multi_doc_chat.utils.model_loader as ml_mod
    from multi_doc_chat.utils import model_loader as ml_mod2

    class FakeApiKeyMgr:
        def __init__(self):
            self.api_keys = {"GROQ_API_KEY": "x", "GOOGLE_API_KEY": "y"}
        def get(self, key: str) -> str: return self.api_keys[key]

    class FakeModelLoader:
        def __init__(self):
            self.api_key_mgr = FakeApiKeyMgr()
            self.config = {"embedding_model": {"model_name": "fake"}, "llm": {"google": {"model_name": "fake"}}}
        def load_embeddings(self): return _StubEmbeddings()
        def load_llm(self): return _StubLLM()

    # Apply patches
    monkeypatch.setattr(ml_mod, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(ml_mod2, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod2, "ModelLoader", FakeModelLoader)
    
    # Also patch imports inside the ingestion module if needed
    try:
        import multi_doc_chat.src.document_ingestion.data_ingestion as di
        monkeypatch.setattr(di, "ModelLoader", FakeModelLoader)
    except ImportError:
        pass # Module might not be imported yet

    yield FakeModelLoader