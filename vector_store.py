import openai
from openai import OpenAI
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from urllib.parse import urlparse
from langchain_community.vectorstores import FAISS

import re
from typing import Generator
from bs4 import BeautifulSoup, Doctype, NavigableString, Tag
import requests
import pickle
from langchain_openai import OpenAIEmbeddings
import os
import json

def langchain_docs_extractor(soup: BeautifulSoup) -> str:
    # Remove all the tags that are not meaningful for the extraction.
    SCAPE_TAGS = ["nav", "footer", "aside", "script", "style"]
    [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]

    def get_text(tag: Tag) -> Generator[str, None, None]:
        for child in tag.children:
            if isinstance(child, Doctype):
                continue

            if isinstance(child, NavigableString):
                yield child
            elif isinstance(child, Tag):
                if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    yield f"{'#' * int(child.name[1:])} {child.get_text()}\n\n"
                elif child.name == "a":
                    yield f"[{child.get_text(strip=False)}]({child.get('href')})"
                elif child.name == "img":
                    yield f"![{child.get('alt', '')}]({child.get('src')})"
                elif child.name in ["strong", "b"]:
                    yield f"**{child.get_text(strip=False)}**"
                elif child.name in ["em", "i"]:
                    yield f"_{child.get_text(strip=False)}_"
                elif child.name == "br":
                    yield "\n"
                elif child.name == "code":
                    parent = child.find_parent()
                    if parent is not None and parent.name == "pre":
                        classes = parent.attrs.get("class", "")

                        language = next(
                            filter(lambda x: re.match(r"language-\w+", x), classes),
                            None,
                        )
                        if language is None:
                            language = ""
                        else:
                            language = language.split("-")[1]

                        lines: list[str] = []
                        for span in child.find_all("span", class_="token-line"):
                            line_content = "".join(
                                token.get_text() for token in span.find_all("span")
                            )
                            lines.append(line_content)

                        code_content = "\n".join(lines)
                        yield f"```{language}\n{code_content}\n```\n\n"
                    else:
                        yield f"`{child.get_text(strip=False)}`"

                elif child.name == "p":
                    yield from get_text(child)
                    yield "\n\n"
                elif child.name == "ul":
                    for li in child.find_all("li", recursive=False):
                        yield "- "
                        yield from get_text(li)
                        yield "\n\n"
                elif child.name == "ol":
                    for i, li in enumerate(child.find_all("li", recursive=False)):
                        yield f"{i + 1}. "
                        yield from get_text(li)
                        yield "\n\n"
                elif child.name == "div" and "tabs-container" in child.attrs.get(
                    "class", [""]
                ):
                    tabs = child.find_all("li", {"role": "tab"})
                    tab_panels = child.find_all("div", {"role": "tabpanel"})
                    for tab, tab_panel in zip(tabs, tab_panels):
                        tab_name = tab.get_text(strip=True)
                        yield f"{tab_name}\n"
                        yield from get_text(tab_panel)
                elif child.name == "table":
                    thead = child.find("thead")
                    header_exists = isinstance(thead, Tag)
                    if header_exists:
                        headers = thead.find_all("th")
                        if headers:
                            yield "| "
                            yield " | ".join(header.get_text() for header in headers)
                            yield " |\n"
                            yield "| "
                            yield " | ".join("----" for _ in headers)
                            yield " |\n"

                    tbody = child.find("tbody")
                    tbody_exists = isinstance(tbody, Tag)
                    if tbody_exists:
                        for row in tbody.find_all("tr"):
                            yield "| "
                            yield " | ".join(
                                cell.get_text(strip=True) for cell in row.find_all("td")
                            )
                            yield " |\n"

                    yield "\n\n"
                elif child.name in ["button"]:
                    continue
                else:
                    yield from get_text(child)

    joined = "\n".join(get_text(soup))
    return re.sub(r"\n\n+", "\n\n", joined).strip()

class VStore:
    # Класс векторное хранилище по определенной базе документов
    # openai_key - API_key от опенАИ - досктуп к модели эмбеддингов от OpenAI, берется из переменной окружения GPT_SECRET_KEY
    # embeddings - модель эмбеддингов
    # name  -  имя файла векторной базы в папке /vector_stores
    # модель gpt (по умолчанию задается в переменной окружения model_gpt
    # chunk_size - размер чанков (в токенах соответствующей модели), по умолчанию 2000
    # db - векторная база (????)
    def __init__(self, name='', model_gpt = ''):   #, chunk_size=0):
        self.embeddings = OpenAIEmbeddings(openai_api_key = os.getenv('GPT_SECRET_KEY'))
        self.name = name
        if model_gpt == '': 
            self.model = os.getenv('model_gpt')
        else:
            self.model = model_gpt
        self.db_dict = {}
        self.db = None
    
    def num_tokens(self, str_):
        # Функция возвращает длину строки  в токенах
        # токенизатор зависит от модели gpt
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(str_))
        return num_tokens
    
    def save(self):
        # Функция сохраняет векторную базу локально в папку /vector_stores, 
        # json - файл со структурой VectorStore без самой базы
        db = self.db
        if db:
            folder_path  = "vector_stores"
            index_name = self.name
            filepath = os.path.join(folder_path, index_name)
            db.save_local(filepath)

            dict_to_save = {
                'name': index_name,
                'chunk_size': self.chunk_size,
                'db_dict': self.db_dict 
                            }
            filename = f'{index_name}.json'
            filepath = os.path.join(folder_path, filename)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(dict_to_save, file, ensure_ascii=False, indent=4)

    def __create_db_dict__(self):
        # функция формирует словарь базы знаний,
        # ключи - url источника, значения - список индексов векторов vector_store
        # 

        dict_ = self.db.docstore._dict
        keys = list(dict_)
        set_urls = set()
        for key in keys:
            doc = dict_.get(key)
            set_urls.add(doc.metadata.get('source'))
        
        db_dict = {}
        for url in set_urls:
            lst_keys = []
            for key in keys:
                doc = dict_.get(key)
                if doc.metadata.get('source') == url:
                    lst_keys.append(key)
            db_dict[url] = lst_keys
        return db_dict

    def __split_doc__(self, text_, url_, chapter_name_):
        # Функция разбивает на чанки длины self.chunk_size text_ со странички wiki по адресу url_, 
        # возвращает список документов лангчейн:
        #   в page_content текстовые чанки, 
        #   в meta_data - заголовки маркдаун, название раздела (odata, ...), url странички

        chunk_list = []
        doc_len = self.num_tokens(text_)
        if doc_len > self.chunk_size:
            headers_to_split_on = [(f"{'#' * i}", f"H{i}") for i in range(1, 5)]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            chunks = markdown_splitter.split_text(text_)
            if len(chunks)>1: chunks = chunks[1:]
            for chunk in chunks:
                headers = ''
                ch_txt = chunk.page_content
                for i in range(1, 5):
                    header_key = f"H{i}" 
                    if header_key in chunk.metadata: headers += f'\n{chunk.metadata[header_key]}'
                pcont = f'{headers}\n{ch_txt}'
                chunk_len = self.num_tokens(pcont)

                if chunk_len > self.chunk_size:
                    r_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n"], 
                                                                chunk_size = self.chunk_size - self.num_tokens(headers), 
                                                                chunk_overlap = 0,
                                                                length_function = lambda x: self.num_tokens(x))
                    parts = r_splitter.split_text(pcont)
                    for k, part in enumerate(parts):
                        if k == 0: # в первом чанке есть заголовки (от маркдаун-чанка)
                            ppcont = part
                        else:
                            ppcont = f'Продолжение {k}{headers}\n{part}'
                        metadata = chunk.metadata
                        metadata['source'] =  url_
                        metadata['chapter'] = chapter_name_
                        metadata['tokens'] =  self.num_tokens(ppcont)
                        metadata['part_num'] =  k+1         
                        chunk_list.append(Document(page_content=ppcont, metadata=metadata))                           
                else: 
                    metadata = chunk.metadata
                    metadata['source'] =  url_
                    metadata['chapter'] = chapter_name_
                    metadata['tokens'] =  chunk_len       
                    chunk_list.append(Document(page_content=pcont, metadata=metadata))
        else:    
            chunk_list.append(Document(page_content=text_, metadata={'source' : url_, 'chapter':chapter_name_, 'tokens': doc_len}))
        return chunk_list

    def create_from_chapter_url(self, chapter_url, chunk_size=0):
        # Функция берет стартовую страницу раздела по chapter_url (сейчас из файла list_to_parse.pkl), 
        # парсит все связанные с ней (вложенные) странички
        # преобразует полученный html-код в текст, размеченный в формате markdawn, 
        # разбивает на чанки длины chunk_size,
        # векторизует их и создает векторное хранилище 
        # возвращает векторное хранилище FAISS

        def get_name(_url):
            from urllib.parse import urlparse
            parsed_url = urlparse(_url)
            path_segments = parsed_url.path.rstrip('/').split('/')
            return path_segments[-1]
        
        if chunk_size==0:
            if self.model == 'gpt-4o': self.chunk_size = 4000
            else: self.chunk_size = 3000   
        else:    
            self.chunk_size = chunk_size

        chapter_name = get_name(chapter_url)

        #@title Чтение списка из ранее сохраненного файла
        try:
            with open('list_to_parse.pkl', 'rb') as f:
                list_to_parse = pickle.load(f)
        except FileNotFoundError:
            print(f"Файл list_to_parse.pkl не существует.")
        except Exception as e:
            print(f"Произошла ошибка: {e}")

#        list_to_parse = [chapter_url]
#        for url in list_to_parse:
#            if url.startswith('/'): url = 'https://python.langchain.com'+url
#            if url.startswith('https://python.langchain.com'):
#                response = requests.get(url)
#                if response.status_code == 200:
#                    bs = BeautifulSoup(response.text, "html.parser")
#                    for a_tag in bs.find_all('a', href=True):
#                        href = a_tag['href']
#                        if href.startswith('/'): href = 'https://python.langchain.com'+href
#                        if href not in list_to_parse and href.startswith('https://python.langchain.com'):
#                            list_to_parse.append(href)            

        #@title Проходим по списку ссылок и из этой странички вынимаем содержимое
        source_chunks = []
        text = ''
        for url in list_to_parse:
            if url.startswith('/'): url = 'https://python.langchain.com'+url
            if url.startswith('https://python.langchain.com'):
                response = requests.get(url)
                if response.status_code == 200:
                    bs = BeautifulSoup(response.text, "html.parser")
                    text = langchain_docs_extractor(bs)
                    source_chunks += self.__split_doc__(text, url, get_name(url))

        self.db = FAISS.from_documents(source_chunks, self.embeddings)
        self.name = chapter_name
        self.db_dict = self.__create_db_dict__()
        self.save()
        return self.db 
    

    def load_vc(self, _name):
        # Функция загружает сохраненную локально векторную базу, 
        # устанавливает свойства из файла _name.json, 
        # возвращает объект VectorStore
        
        folder_path  = "vector_stores"
        index_name = _name
        index_path = os.path.join(folder_path, index_name)
        json_path = os.path.join(folder_path, f'{index_name}.json')
        self.name = _name

        try:
            self.db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            with open(json_path, 'r', encoding='utf-8') as file:
                json_dict = json.load(file)
            if 'chunk_size' in json_dict: self.chunk_size = json_dict.get('chunk_size')
            if 'db_dict' in json_dict: self.db_dict = json_dict.get('db_dict')
        except:
            print('Не удалось загрузить VS from: ', index_path)

        return self
    
