from urllib.parse import urlparse
from langchain_core.documents import Document

from langchain_community.document_loaders import RecursiveUrlLoader
from utils.parser import docs_extractor

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
import re
from IPython.display import Markdown
import io
import json
import tiktoken
import requests


class UniversalLoader:
    """UniversalLoader contains collection of 'load_from_X' methods to load documents from different sources.'"""

    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def load_from_urls(self, urls: list[str], recursive=False, max_depth=8, timeout=60, header_level=3, **kwargs) -> list[Document]:
        """
        Parses provided URLs and return documents.

        :param header_level:
        :param urls: URLS to parse.
        :param recursive: Parse recursive starting from URLs.
        :param max_depth: Maximum depth of recursion.
        :param timeout: Page loading timeout.
        :param kwargs: Minimal header level to split on.
        :return: List of documents.
        """

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[('#' * i, f'h{i}') for i in range(1, header_level + 1)])

        documents = []
        for url in urls:
            loader = RecursiveUrlLoader(
                url,
                max_depth=max_depth if recursive else 1,
                extractor=docs_extractor,
                timeout=timeout,
                prevent_outside=True,
                check_response_status=True,
                continue_on_failure=True,
                encoding=self.encoding
            )
            docs = loader.load()

            # (Optional splits, checks goes here)
            doc_parts = []
            for doc in docs:
                splits = splitter.split_text(doc.page_content)
                for sp in splits:
                    if sp.metadata:
                        doc_parts.append(
                            Document(
                                page_content=sp.page_content,
                                metadata=doc.metadata | sp.metadata
                            )
                        )
            documents.extend(doc_parts)
        return documents

    def load_ipynb(self,ipynblinklist):
        # декодировщик страшных символов, которые попадаются в нотбуках
        # может такие символы и нормально тоже ищутся в фаис(не проверял), но смотреть в ячейках такой текст невозможно
        def decode_unicode_escapes(text):
            # Поиск символов вида \\u0412\\u044B
            unicode_pattern = re.compile(r'\\u[0-9A-Fa-f]{4}')
            # Замена символов вида \\u0412\\u044B на их раскодированные значения
            decoded_text = unicode_pattern.sub(lambda x: x.group().encode().decode('unicode_escape'), text)
            return decoded_text

        def recursive_decode(data):
            if isinstance(data, str):
                return decode_unicode_escapes(data)
            elif isinstance(data, list):
                return [recursive_decode(item) for item in data]
            elif isinstance(data, dict):
                return {key: recursive_decode(value) for key, value in data.items()}
            else:
                return data

        def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:
            """Возвращает количество токенов в строке"""
            encoding = tiktoken.get_encoding(encoding_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens

        # функция для загрузки блокнота ipynb в Document LangChain
        # считать и преобразовать в текст, выкинуть оутпут ячейки и картинки
        def loadipynb(ipynblink):
            # Регулярное выражение для извлечения file_id
            file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)|/drive/([a-zA-Z0-9_-]+)', ipynblink)
            file_id = file_id_match.group(1) if file_id_match.group(1) else file_id_match.group(2)

            # Формирование прямой ссылки для скачивания файла
            download_link = f"https://drive.google.com/uc?export=download&id={file_id}"

            # Загрузка файла
            response = requests.get(download_link)
            response.raise_for_status()  # Проверка на ошибки

            notebook_content = io.BytesIO(response.content)
            notebook = json.load(notebook_content)

            codeANDmarkdown_cells=[]
            codeANDmarkdown_cells.append('startsection')
            prev_cell_type = None  # Переменная для отслеживания типа предыдущей ячейки
            prevstart=0
            startind=0
            maxchank=0
            # Просматриваем код и маркдаун, а ячейки вывода игнорируем

            for i,cell in enumerate(notebook['cells']):
                nextpeace=""
                if cell['cell_type'] == 'code':
                    # В ячейках кода попадаются страшные символы - раскодировать их
                    nextpeace='\n'.join(recursive_decode(cell['source']))
                elif cell['cell_type'] == 'markdown':
                    nextpeace='\n'.join(cell['source'])
                if "data:image/jpeg" in nextpeace:
                    # это картинка не добавлять в индекс
                    # но вывести, вдруг там что нужное попалось
                    # print(f"эта ячейка пропущена, тщательно проверить, что только картинку пропустили при индексации \n{cell}")
                    # display(Markdown(f"эта ячейка пропущена, тщательно проверить, что только картинку пропустили при индексации \n{cell}"))
                    continue
                if cell['cell_type'] == 'code':
                    # В ячейках кода попадаются страшные символы - раскодировать их
                    codeANDmarkdown_cells.append(nextpeace)
                    prev_cell_type = 'code'  # Обновляем тип предыдущей ячейки
                elif cell['cell_type'] == 'markdown':
                    startind=i
                    # Проверяем, является ли текущая ячейка markdown первой или перед ней была ячейка code
                    if prev_cell_type is None or prev_cell_type == 'code':
                        # для MarkdownHeaderTextSplitter проставляем теги, по которым делить на чанки.
                        # предположительно считать чанком маркдаун(1 или несколько подряд идущих) и далее ячейки с кодом(1 или несколько подряд идущих)
                        # а когда опять попался маркдаун - это будет начало другого чанка
                        codeANDmarkdown_cells.append('startsection')
                        # отследить размер предыдущего чанка
                        if prevstart!=startind:
                            # измерить размер чанка
                            # chanksize=sum(len(s) for s in codeANDmarkdown_cells[prevstart:startind])
                            chanksize=num_tokens_from_string("".join(codeANDmarkdown_cells[prevstart:startind]))
                            if maxchank<chanksize:
                                maxchank=chanksize
                            prevstart=startind
                    codeANDmarkdown_cells.append(nextpeace)
                    prev_cell_type = 'markdown'  # Обновляем тип предыдущей ячейки

            text='\n'.join(codeANDmarkdown_cells)
            return text,maxchank

        # поделить текст на куски
        def splitipynbtext(text,ipynblink):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000, #размер не важен, делится только по одному сепаратору startsection
                chunk_overlap=0,
                length_function=lambda x: num_tokens_from_string(x),
                # не предполагается делить на более мелкие чанки, чем по "startsection"
                separators=["startsection"]
            )
            sublist=[]
            for i,chunk in enumerate(splitter.split_text(text)):
                # удаление разделителей
                cleaned_text = re.sub(r'startsection', '', chunk)
                # сохранить адресацию исходного документа и порядковые номера кусочков
                metadata = {
                    'subid':i,
                    'total':0,
                    'source': ipynblink
                }
                readydoc=Document(page_content=cleaned_text, metadata=metadata)
                sublist.append(readydoc)
            for chunk in sublist:
                chunk.metadata['total']=len(sublist)
            return sublist

        text,maxchank=loadipynb(ipynblinklist[0])
        # print(f"Максимальный размер чанка в токенах: {maxchank}")
        ipynbdocs=splitipynbtext(text,ipynblinklist[0])
        return ipynbdocs

    # load_from_X methods ...
