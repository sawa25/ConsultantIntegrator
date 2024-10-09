from uuid import uuid4
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.indexes import SQLRecordManager, index

from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv


load_dotenv()


def create_engine(store_engine, embeddings, dimensions, **kwargs):
    if store_engine.__name__ == 'FAISS':
        # See https://github.com/langchain-ai/langchain/discussions/13773+
        return FAISS(
            embedding_function=embeddings,
            index=IndexFlatL2(dimensions),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
    else:
        raise NotImplementedError('Not implemented yet')


class VectorStore:
    """VectorStore class is wrapper for vector storages which supports indexing.'"""
    def __init__(
            self,
            embedding=OpenAIEmbeddings(),
            description='',
            dimensions=1536,
            source_id_key="source"
    ):
        self.description = description
        self.location = None
        self.store_engine = FAISS
        self.embeddings = embedding
        self.dimensions = dimensions
        self.vector_store = create_engine(self.store_engine, self.embeddings, self.dimensions)

        self.source_id_key = source_id_key
        self.record_manager = SQLRecordManager(
            namespace=self.store_engine.__name__,
            db_url='sqlite:///:memory:'
        )
        self.record_manager.create_schema()

    @classmethod
    def from_documents(cls, documents: list[Document] = [],
                       embedding=OpenAIEmbeddings(), **kwargs):
        store = cls(embedding, **kwargs)
        store.add_documents(documents)
        return store

    def add_documents(self, documents: list[Document] = [], **kwargs):
        return self._index(documents, cleanup="incremental")

    def update_documents(self, documents: list[Document] = [], delete=False, **kwargs):
        return self._index(documents, cleanup='full' if delete else 'incremental')

    def merge_from(self, other):
        #self.vector_store.merge_from(other.vector_store)
        #self.record_manager.update(other.record_manager.list_keys())
        documents = list(other.vector_store.docstore._dict.values())
        self.add_documents(documents)

    def clear(self):
        """Hacky helper method to clear content."""
        if self.vector_store is not None:
            self._index(cleanup='full')

    def search(self, query: str, **kwargs):
        return self.vector_store.search(query, **kwargs)

    def save(self, save_path: str | Path):
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)

        self.vector_store.save_local(save_path)
        with Path(save_path / 'description.md').open('w') as f:
            f.write(self.description)

        # create new manager and copy records
        if self.location != save_path:
            record_manager = SQLRecordManager(
                namespace=self.record_manager.namespace,
                db_url=f"sqlite:///{str(save_path / 'cache.sql')}"
            )
            record_manager.create_schema()
            record_manager.update(self.record_manager.list_keys())
            self.record_manager = record_manager
        
        self.location = save_path

    def load(self, save_path: str | Path):
        save_path = Path(save_path)
        self.vector_store = self.store_engine.load_local(
            save_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        with Path(save_path / 'description.md').open() as f:
            self.description = f.read()
            
        self.record_manager = SQLRecordManager(
            namespace=self.store_engine.__name__,
            db_url=f"sqlite:///{str(save_path / 'cache.sql')}"
        )
        self.location = save_path

    def _index(self, documents: list[Document] = [], cleanup=None) -> dict:
        return index(
            documents,
            self.record_manager,
            self.vector_store,
            cleanup=cleanup if cleanup else self.cleanup,
            source_id_key=self.source_id_key
        )


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    embeddings = OpenAIEmbeddings()
    doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
    doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})
    store = VectorStore.from_documents([doc1, doc2], embedding=embeddings)
