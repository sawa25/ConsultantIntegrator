from consultant import VectorStore
from consultant import UniversalLoader

loader = UniversalLoader()
docs = loader.load_from_urls(
    ['https://python.langchain.com/v0.2/docs/introduction/']
)

store = VectorStore.from_documents(docs, name='test')
store.save()
