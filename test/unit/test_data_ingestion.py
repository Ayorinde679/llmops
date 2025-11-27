import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import json
import hashlib
import sys

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor, FaissManager, generate_session_id, DocumentPortalException


# --- Fixtures and Mocks Setup ---

@pytest.fixture
def mock_model_loader():
    """Mocks the ModelLoader and its load_embeddings method."""
    mock_loader = MagicMock()
    mock_embeddings = MagicMock(spec=['embed_documents'])
    mock_loader.load_embeddings.return_value = mock_embeddings
    return mock_loader

@pytest.fixture
def mock_faiss():
    """Mocks the FAISS VectorStore class and instance methods."""
    mock_faiss_class = MagicMock()
    mock_faiss_instance = MagicMock(spec=['add_documents', 'save_local', 'as_retriever'])
    mock_faiss_class.load_local.return_value = mock_faiss_instance
    mock_faiss_class.from_texts.return_value = mock_faiss_instance
    return mock_faiss_class, mock_faiss_instance

@pytest.fixture
def chat_ingestor_setup(tmp_path, mock_model_loader):
    """Initializes ChatIngestor with mocked dependencies and temp directories."""
    temp_base = tmp_path / "data"
    faiss_base = tmp_path / "faiss_index"

    with patch('document_ingestor.ModelLoader', return_value=mock_model_loader):
        with patch('document_ingestor.generate_session_id', return_value="test_session"):
            ingestor = ChatIngestor(
                temp_base=str(temp_base),
                faiss_base=str(faiss_base),
                use_session_dirs=True
            )
            return ingestor, temp_base, faiss_base

@pytest.fixture
def faiss_manager_setup(tmp_path, mock_model_loader):
    """Initializes FaissManager with mocked dependencies and temp directory."""
    index_dir = tmp_path / "faiss_test_index"
    index_dir.mkdir()
    
    with patch('document_ingestor.ModelLoader', return_value=mock_model_loader):
        manager = FaissManager(index_dir, mock_model_loader)
        return manager, index_dir

# --- ChatIngestor Tests ---

## Initialization and Utility Tests

def test_generate_session_id_format():
    """Tests the format of the session ID."""
    session_id = generate_session_id()
    parts = session_id.split('_')
    assert len(parts) == 3
    assert parts[0] == "session"
    assert len(parts[2]) == 8

def test_chat_ingestor_initialization_sessionized(chat_ingestor_setup):
    """Tests sessionized initialization and directory creation."""
    ingestor, temp_base, faiss_base = chat_ingestor_setup
    
    assert ingestor.session_id == "test_session"
    assert ingestor.temp_dir == temp_base / "test_session"
    assert ingestor.faiss_dir == faiss_base / "test_session"
    assert ingestor.temp_dir.is_dir()
    assert ingestor.faiss_dir.is_dir()

def test_chat_ingestor_initialization_non_sessionized(tmp_path, mock_model_loader):
    """Tests non-sessionized initialization where base directories are used directly."""
    temp_base = tmp_path / "data_base"
    faiss_base = tmp_path / "faiss_base"

    with patch('document_ingestor.ModelLoader', return_value=mock_model_loader):
        ingestor = ChatIngestor(
            temp_base=str(temp_base),
            faiss_base=str(faiss_base),
            use_session_dirs=False
        )
        
        assert ingestor.temp_dir == temp_base
        assert ingestor.faiss_dir == faiss_base

@patch('document_ingestor.RecursiveCharacterTextSplitter')
def test_chat_ingestor_split_documents(MockSplitter, chat_ingestor_setup):
    """Tests the document splitting logic by checking parameters and call count."""
    ingestor, _, _ = chat_ingestor_setup
    mock_docs = [MagicMock()] * 3
    
    mock_instance = MockSplitter.return_value
    mock_instance.split_documents.return_value = [MagicMock()] * 7

    chunks = ingestor._split(mock_docs, chunk_size=700, chunk_overlap=150)
    
    MockSplitter.assert_called_once_with(chunk_size=700, chunk_overlap=150)
    mock_instance.split_documents.assert_called_once_with(mock_docs)
    assert len(chunks) == 7

## Retriever Building Tests

@patch('document_ingestor.FaissManager')
@patch('document_ingestor.ChatIngestor._split')
@patch('document_ingestor.load_documents', return_value=[MagicMock()] * 2)
@patch('document_ingestor.save_uploaded_files', return_value=["file1"])
def test_built_retriever_success_mmr(
    mock_save_files, mock_load_docs, mock_split, MockFaissManager, chat_ingestor_setup, mock_faiss
):
    """Tests successful retriever building using MMR search type and its parameters."""
    ingestor, _, _ = chat_ingestor_setup
    mock_faiss_class, mock_faiss_instance = mock_faiss
    
    mock_chunks = [
        MagicMock(page_content="c1", metadata={"source": "d1"}),
        MagicMock(page_content="c2", metadata={"source": "d2"}),
    ]
    mock_split.return_value = mock_chunks
    
    mock_fm_instance = MockFaissManager.return_value
    mock_fm_instance.load_or_create.return_value = mock_faiss_instance
    mock_fm_instance.add_documents.return_value = 2
    
    with patch('document_ingestor.FAISS', new=mock_faiss_class):
        
        ingestor.built_retriver(
            uploaded_files=MagicMock(),
            search_type="mmr",
            k=3,
            fetch_k=10,
            lambda_mult=0.8
        )
        
        mock_fm_instance.add_documents.assert_called_once_with(mock_chunks)

        mock_faiss_instance.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.8}
        )

@patch('document_ingestor.FaissManager')
@patch('document_ingestor.load_documents', return_value=[MagicMock()] * 2)
@patch('document_ingestor.save_uploaded_files', return_value=["file1"])
def test_built_retriever_success_similarity(
    mock_save_files, mock_load_docs, MockFaissManager, chat_ingestor_setup, mock_faiss
):
    """Tests successful retriever building using similarity search type."""
    ingestor, _, _ = chat_ingestor_setup
    mock_faiss_class, mock_faiss_instance = mock_faiss
    
    mock_fm_instance = MockFaissManager.return_value
    mock_fm_instance.load_or_create.return_value = mock_faiss_instance
    
    with patch.object(ingestor, '_split', return_value=[MagicMock()]*2):
        with patch('document_ingestor.FAISS', new=mock_faiss_class):
            
            ingestor.built_retriver(
                uploaded_files=MagicMock(),
                search_type="similarity",
                k=5
            )
            
            mock_faiss_instance.as_retriever.assert_called_once_with(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

@patch('document_ingestor.load_documents', return_value=[])
@patch('document_ingestor.save_uploaded_files', return_value=["file1"])
def test_built_retriever_no_valid_documents_error(mock_save_files, mock_load_docs, chat_ingestor_setup):
    """Tests error handling when document loading returns an empty list."""
    ingestor, _, _ = chat_ingestor_setup
    with pytest.raises(DocumentPortalException) as excinfo:
        ingestor.built_retriver(uploaded_files=MagicMock())
    
    assert "Failed to build retriever" in str(excinfo.value)
    assert "No valid documents loaded" in str(excinfo.value.__cause__)

# --- FaissManager Tests ---

def test_faiss_manager_initialization(faiss_manager_setup):
    """Tests FaissManager initialization and meta file loading/creation."""
    manager, index_dir = faiss_manager_setup
    
    assert manager._meta == {"rows": {}}
    
    dummy_meta = {"rows": {"key1": True}}
    manager.meta_path.write_text(json.dumps(dummy_meta))
    
    new_manager = FaissManager(index_dir, manager.model_loader)
    assert new_manager._meta == dummy_meta

def test_faiss_manager_fingerprint(faiss_manager_setup):
    """Tests the _fingerprint de-duplication logic."""
    manager, _ = faiss_manager_setup
    text = "document content"

    meta1 = {"source": "/path/file.txt"}
    fp1 = manager._fingerprint(text, meta1)
    assert fp1 == "/path/file.txt::"

    meta2 = {"source": "/path/file.txt", "row_id": 5}
    fp2 = manager._fingerprint(text, meta2)
    assert fp2 == "/path/file.txt::5"

    meta3 = {}
    fp3 = manager._fingerprint(text, meta3)
    assert len(fp3) == 64
    assert fp3 == hashlib.sha256(text.encode("utf-8")).hexdigest()

@patch('document_ingestor.FAISS')
def test_faiss_manager_load_or_create_load(MockFAISS, faiss_manager_setup):
    """Tests loading an existing FAISS index when files exist."""
    manager, index_dir = faiss_manager_setup
    
    (index_dir / "index.faiss").touch()
    (index_dir / "index.pkl").touch()
    
    mock_vs = MagicMock()
    MockFAISS.load_local.return_value = mock_vs

    manager.load_or_create(texts=["ignored"], metadatas=[{}])
    
    MockFAISS.load_local.assert_called_once()

@patch('document_ingestor.FAISS')
def test_faiss_manager_load_or_create_create(MockFAISS, faiss_manager_setup):
    """Tests creating a new FAISS index when files do not exist."""
    manager, index_dir = faiss_manager_setup
    
    texts = ["text1", "text2"]
    metadatas = [{"source": "a"}, {"source": "b"}]
    
    mock_vs = MagicMock()
    MockFAISS.from_texts.return_value = mock_vs
    
    manager.load_or_create(texts=texts, metadatas=metadatas)
    
    MockFAISS.from_texts.assert_called_once()
    mock_vs.save_local.assert_called_once_with(str(index_dir))

@patch('document_ingestor.FAISS')
def test_faiss_manager_add_documents_idempotency(MockFAISS, faiss_manager_setup):
    """Tests adding documents, verifying that duplicates are skipped using the fingerprint."""
    manager, _ = faiss_manager_setup
    
    mock_vs = MagicMock(spec=['add_documents', 'save_local', 'as_retriever'])
    MockFAISS.from_texts.return_value = mock_vs
    manager.load_or_create(texts=["initial"], metadatas=[{}])

    doc_new_1 = MagicMock(page_content="Unique chunk 1", metadata={"source": "file_b.pdf"})
    doc_new_2 = MagicMock(page_content="Unique chunk 2", metadata={"source": "file_c.pdf"})
    doc_duplicate = MagicMock(page_content="Unique chunk 1", metadata={"source": "file_b.pdf"}) 

    docs_to_add = [doc_new_1, doc_new_2, doc_duplicate]
    
    key_for_doc_new_1 = manager._fingerprint(doc_new_1.page_content, doc_new_1.metadata)
    manager._meta["rows"][key_for_doc_new_1] = True
    
    added_count = manager.add_documents(docs_to_add)

    assert added_count == 1
    
    mock_vs.add_documents.assert_called_once()
    added_docs_call = mock_vs.add_documents.call_args[0][0]
    assert len(added_docs_call) == 1
    assert added_docs_call[0] == doc_new_2

    mock_vs.save_local.assert_called_once()

def test_faiss_manager_add_documents_no_load_error(faiss_manager_setup):
    """Tests RuntimeError when add_documents is called before load_or_create."""
    manager, _ = faiss_manager_setup
    
    with pytest.raises(RuntimeError) as excinfo:
        manager.add_documents([MagicMock()])
        
    assert "Call load_or_create() before add_documents_idempotent()" in str(excinfo.value)