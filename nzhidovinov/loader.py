from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders import RecursiveUrlLoader
from utils.parser import docs_extractor


class UniversalLoader:
    """UniversalLoader contains collection of 'load_from_X' methods to load documents from different sources.'"""

    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def load_from_urls(self, urls: list[str] = [], recursive=False, **kwargs) -> list[Document]:
        """
        Parses provided URLs and return documents.

        :param urls: URLS to parse.
        :param recursive: Parse recursive starting from URLs.
        :param kwargs:
        :return: List of documents.
        """

        def get_name(_url):
            path_segments = urlparse(_url).path.rstrip('/').split('/')
            return path_segments[-1]

        def get_base_url(_url: str):
            parts = _url.split('/')
            return f'{parts[0]}//{parts[2]}'

        documents = []
        for url in urls:
            loader = RecursiveUrlLoader(
                url,
                max_depth=8 if recursive else 1,
                extractor=docs_extractor,
                timeout=60,
                prevent_outside=True,
                check_response_status=True,
                continue_on_failure=True,
                encoding=self.encoding
            )
            docs = loader.load()
            # (Optional splits, checks goes here)
            documents.extend(docs)

        return documents

    # load_from_X methods ...from
