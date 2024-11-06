import os
from typing import Annotated, Literal, List
from typing_extensions import TypedDict

from langchain.docstore.document import Document
from langchain.schema import Document

from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
import os

from typing import List

load_dotenv()

GPT_SECRET_KEY = os.environ.get("GPT_SECRET_KEY")
os.environ["OPENAI_API_KEY"] = GPT_SECRET_KEY
Tavily = os.environ.get("Tavily")
os.environ["TAVILY_API_KEY"] = Tavily

temperature = 0.1
model_name ="gpt-4o-mini" 
embd = OpenAIEmbeddings()


db_name = 'lesson_video_notes_2'         #@param ["langchain_openai_1536", "lesson_video_notes_2","lessons_notebook", "less_video"]
db_name_lan = 'langchain_openai_1536'    #@param ["langchain_openai_1536", "lesson_video_notes_2","lessons_notebook", "less_video"]


db_folder_path = "vector_stores" # todo
full_path = os.path.join(db_folder_path, db_name)
full_path_lan = os.path.join(db_folder_path, db_name_lan)
num_docs_for_search = 6 #@param {type: "slider", min: 1, max: 15, step:1}
db = FAISS.load_local(full_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": num_docs_for_search})
db_lan = FAISS.load_local(full_path_lan, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever_lan = db_lan.as_retriever(search_kwargs={"k": num_docs_for_search})
num_links_for_TavilySearch = 2 #@param {type: "slider", min: 1, max: 8, step:1}


llm = ChatOpenAI(model= model_name)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        messages: list of messages
        baza: baza
    """

    question: str
    generation: str
    documents: List[str]
    messages: Annotated[list, add_messages]
    baza: str


graph_builder = StateGraph(GraphState)

### Router (маршрутизатор вопроса пользователя: в БД или в интернет)
# Data model
class RouteQuery(BaseModel):
    """Направляет запрос пользователя к наиболее подходящему источнику данных."""

    datasource: Literal["vectorstore", "web_search", "without_context", 'langchain'] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore or a without_context or a langchain.",
    )
# В зависимости от вопроса пользователя выберите, направить его на веб-поиск или в векторное хранилище.

# LLM with function call
llm = ChatOpenAI(model=model_name, temperature=temperature)
structured_llm_router = llm.with_structured_output(RouteQuery)


system ='''
You are an expert at routing user questions to either the vectorstore, web search, or langchain. Follow these guidelines:

Vectorstore:

Use for questions about AI University lessons on neural assistant development (including neural sales assistant, neural content creator, neural quality control, etc.)
Python programming
Embedding creation
Prompt engineering


Web search:

Use for any questions requiring current or real-time information
This includes recent updates or news about topics that might otherwise fall under vectorstore or langchain
Always prioritize web search for questions explicitly asking about "current", "latest", or "up-to-date" information on any topic


Langchain:

Use for questions specifically about langchain functionality, usage, or concepts
Do not use for questions about recent updates or news related to langchain (use web search instead)


Without context:

Use for any questions that don't fit the above categories



Remember: If there's any doubt about whether information needs to be current, always default to web search.
'''

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


### Retrieval Grader (оценка соответствия извлеченных из хранилища документов/чанков вопросу)

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(...,
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM with function call
llm = ChatOpenAI(model=model_name, temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

### Generate (генерация ответа)

system = '''
You are an advanced AI assistant designed for comprehensive question-answering and task completion. Your responses should be:

Thorough and informative, providing in-depth explanations without arbitrary length restrictions.
Tailored to the user's level of expertise, adjusting complexity as needed.
Supported by relevant examples, analogies, or case studies when appropriate.
Backed by code snippets (preferably in Python) for programming-related queries, including detailed comments and explanations.
Structured logically, using markdown formatting for enhanced readability (e.g., headings, lists, code blocks).
Mindful of potential biases or limitations in the provided information.

When faced with incomplete information:

Clearly state what is known and unknown.
Offer educated assumptions or hypotheses, clearly labeled as such.
Suggest additional resources or avenues for further research.

For complex queries:

Break down the problem into manageable steps.
Explain your reasoning process thoroughly.
Provide alternative approaches or solutions when applicable.

Always prioritize accuracy over speed. If a task requires extensive computation or research, outline the approach you 
would take and offer to proceed step-by-step with user confirmation.
Continuously seek clarification and feedback to ensure your responses meet the user's needs. 
Be prepared to iterate and refine your answers based on follow-up questions.
Maintain a professional yet engaging tone, and adapt your language style to the context of 
the conversation (e.g., formal for business queries, more casual for creative tasks).
Remember to respect ethical guidelines and intellectual property rights in your responses.
Answer in Russian.
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: \n\n {question} \n\n Context: {context}"),
    ]
)

# LLM
llm = ChatOpenAI(model_name=model_name, temperature=temperature)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

### Answer Grader (оценка соответствия ответа вопросу)

# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(...,
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model=model_name, temperature=temperature)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""


answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader


### Question Re-writer  (преобразователь исходных вопросов в улученную версию, оптимизированную для поиска в векторном хранилище)

# LLM
llm = ChatOpenAI(model=model_name, temperature=temperature)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. Reply in Russian"""



re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

### Hallucination Grader  (оценка уровня галюцинации агента по ответам - соответствует ли сгенерированный ответ фактам, содержащимся в чанках)

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model=model_name, temperature=temperature)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
# hallucination_grader.ainvoke({"documents": docs, "generation": generation})

### Search WEB-searche tool (определяем инструмент для поиска информации в интернете
web_search_tool = TavilySearchResults(k=num_links_for_TavilySearch)



##Construct the Graph (конструируем Граф)
###Определим состояния

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        messages: list of messages
        baza: baza
    """

    question: str
    generation: str
    documents: List[str]
    messages: Annotated[list, add_messages]
    baza: str

### Define Graph Flow (определим структуру графа: узлы/ребра)
def format_history(messages):
    """
    Преобразование сообщений в строку для использования в prompt.
    """
    history_str = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history_str += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"AI: {msg.content}\n"
    return history_str


# Nodes (узлы)
async def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
   
    question = state.get("question", "Нет вопроса")
    #state["baze"] = 'vectorstore'

    # Retrieval
    documents = await retriever.ainvoke(question)

    return {"documents": documents, "question": question, "baza": "vectorstore"}

async def retrieve_lan(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
   
    question = state.get("question", "Нет вопроса")
    #state["baze"] = "langchain"
    # Retrieval
    documents = await retriever_lan.ainvoke(question)

    return {"documents": documents, "question": question, "baza": "langchain"}

async def generate(state):
    """
    Generate answer
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    question = state.get("question", "Нет вопроса")
    #generation = state.get("generation", "Нет ответа")
    documents = state.get("documents",[])
    baza = state.get("baza",'')

    # RAG generation
    generation = await rag_chain.ainvoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, 
            "messages": [HumanMessage(content=question), AIMessage(content=generation)], "baza":baza}

async def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """


    question = state.get("question", "Нет вопроса")
    documents = state.get("documents",[])
    baza = state.get("baza", "")

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = await retrieval_grader.ainvoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":

            filtered_docs.append(d)
        else:
            continue
    #return {"documents": filtered_docs, "question": question}
    return {"documents": filtered_docs, "question": question, "baza": baza}

async def transform_query(state):
    """
    Transform the query to produce a better question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    question = state.get("question", "Нет вопроса")
    documents = state.get("documents",[])
    baza = state.get("baza","blank")

    # Re-write question
    better_question = await question_rewriter.ainvoke({"question": question})
    #return {"documents": documents, "question": better_question}
    return {"documents": documents, "question": better_question, "baza": baza}

async def web_search(state): ####!!!!&&&&&&&&&?????????
    """
    Web search based on the re-phrased question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state.get("question", "Нет вопроса")

    # Web search
    docs = await web_search_tool.ainvoke({"query": question}) ####!!!!
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

# нода - поболтать
async def chatbot(state):

    question = state.get("question", "Нет вопроса")
    documents = state.get("documents", "Нет контекста")
    dia_history = format_history(state.get("messages", []))

    system = '''You are an assistant for question-answering tasks.
    Take into account the entire conversation history when answering, to maintain context and provide coherent responses.
    '''
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", system),
        ("human", "Question: \n\n {question} \n\n Context: {context} \n\n Dialog's history: {history}"),
      ]
    )
    messages = prompt.format_messages(context=documents, question=question, history=dia_history)

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    generation = await llm.ainvoke(messages)

    response = generation.content if hasattr(generation, 'content') else str(generation)

    return {"question": question, "generation": response, "messages": [HumanMessage(content=question), AIMessage(content=response)]}

def blank(state):
  new_message = [AIMessage(content="Вы задали некорректный вопрос, уточните, пожалуйста, чем я могу Вам помочь.")]
  return {"messages": new_message}


### Edges (ребра) ###

async def route_question(state):
    """
    Route question to web search or RAG or ChatBot.
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """

    question = state["question"]
    source = await question_router.ainvoke({"question": question})

    if source.datasource == "web_search":

        return "web_search"
    
    elif source.datasource == "vectorstore":

        return "vectorstore"
    
    elif source.datasource == "langchain":

        return "langchain"
    else:

        return "without_context"
    

def route_after_transform(state):

    return state.get("baza", "blank")


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """


    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        return "transform_query"
    else:
        # We have relevant documents, so generate answer\
        return "generate"

async def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Decision for next node to call
    """


    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = await hallucination_grader.ainvoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        # Check question-answering
       
        score = await answer_grader.ainvoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
    
            return "useful"
        else:
    
            return "not useful"
    else:
            return "not supported"



#from inspect import EndOfBlock - Compile Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("retrieve_lan", retrieve_lan)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("chatbot", chatbot)  # chatbot (поболтать)
workflow.add_node("blank", blank)  # загрушка: отчет без GPT

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "langchain":"retrieve_lan",
        "without_context": "chatbot",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
#workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
      "transform_query",
      route_after_transform,
      {
        "vectorstore": "retrieve",
        "langchain":"retrieve_lan",
        "blank":"blank"}
                )
workflow.add_edge("chatbot", END) # мое
workflow.add_edge("blank", END)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "chatbot",   # мое
        "useful": END,
        "not useful": "transform_query",
    },
)

memory = MemorySaver()
# Compile
app = workflow.compile(checkpointer=memory)
