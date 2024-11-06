import Graph_builder

import os
from typing import  List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
import os
import re

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from dotenv import load_dotenv


from telegram import Bot, Message
from telegram.constants import ParseMode, ChatAction
import asyncio

import telegram.error
from typing import Optional, List, Tuple

load_dotenv()
TOKEN = os.environ.get("TOKEN")


key_name = 'name'
key_cont = 'gpt_context'
# temperature = 0.1
# model_name ="gpt-4o-mini" 


#from inspect import EndOfBlock - Compile Graph
workflow = StateGraph(Graph_builder.GraphState)

# Define the nodes
workflow.add_node("web_search", Graph_builder.web_search)  # web search
workflow.add_node("retrieve", Graph_builder.retrieve)  # retrieve
workflow.add_node("retrieve_lan", Graph_builder.retrieve_lan)  # retrieve
workflow.add_node("grade_documents", Graph_builder.grade_documents)  # grade documents
workflow.add_node("generate", Graph_builder.generate)  # generate
workflow.add_node("transform_query", Graph_builder.transform_query)  # transform_query
workflow.add_node("chatbot", Graph_builder.chatbot)  # chatbot (поболтать)
workflow.add_node("blank", Graph_builder.blank)  # загрушка: отчет без GPT

# Build graph
workflow.add_conditional_edges(
    START,
    Graph_builder.route_question,
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
    Graph_builder.decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
#workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
      "transform_query",
      Graph_builder.route_after_transform,
      {
        "vectorstore": "retrieve",
        "langchain":"retrieve_lan",
        "blank":"blank"}
                )
workflow.add_edge("chatbot", END) # мое
workflow.add_edge("blank", END)
workflow.add_conditional_edges(
    "generate",
    Graph_builder.grade_generation_v_documents_and_question,
    {
        "not supported": "chatbot",   # мое
        "useful": END,
        "not useful": "transform_query",
    },
)

memory = MemorySaver()
# Compile
graph = workflow.compile(checkpointer=memory)

#################################################################################################################                 
# Эта функция запускает граф на исполнение
async def run_graph(user_input, user_id):
    config = {"configurable": {"thread_id": user_id}}

    async for event in graph.astream({"messages": [("user", user_input)], "question": user_input}, config):
        for value in event.values():
            None       
    return value["generation"]
#################################################################################################################                 
# Функция отправки длинного сообщения постепенно, как будто бот печатает его
async def send_message_gradually(bot: Bot, chat_id: int, text: str, message_id: int, delay: float = 0.2, chunk_size: int = 50):
    """
    Отправляет длинное сообщение постепенно, с эффектом набора текста, без разрыва слов.
    
    Args:
        bot: Объект бота Telegram
        chat_id: ID чата
        text: Текст для отправки
        message_id: ID сообщения для редактирования
        delay: Задержка между частями сообщения (в секундах)
        chunk_size: Примерный размер каждой части
    """
    # Определяем константы в начале функции
    MAX_MESSAGE_LENGTH = 4096
    
    def smart_text_split(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
        """
        Разделяет текст на части по смысловым блокам.
        """
        header_pattern = r'^#{1,6}\s+.+$'
        list_item_pattern = r'^\s*[-*]\s+.+$'
        
        lines = text.split('\n')
        messages: List[str] = []
        current_message: List[str] = []
        current_length = 0
        
        def is_header(line: str) -> bool:
            return bool(re.match(header_pattern, line.strip()))
        
        def is_list_item(line: str) -> bool:
            return bool(re.match(list_item_pattern, line.strip()))
        
        def get_block_end(lines: List[str], start: int) -> int:
            if start >= len(lines):
                return start
            
            current_line = lines[start].strip()
            
            if is_header(current_line):
                for i in range(start + 1, len(lines)):
                    if is_header(lines[i].strip()):
                        return i
                return len(lines)
            
            if is_list_item(current_line):
                last_list_item = start
                for i in range(start + 1, len(lines)):
                    if is_list_item(lines[i].strip()):
                        last_list_item = i
                    elif lines[i].strip() and not lines[i].strip().startswith(('  ', '\t')):
                        break
                return last_list_item + 1
            
            in_paragraph = bool(current_line)
            for i in range(start + 1, len(lines)):
                line = lines[i].strip()
                if in_paragraph:
                    if not line or is_header(line) or is_list_item(line):
                        return i
                else:
                    if line:
                        return i
            return len(lines)

        i = 0
        while i < len(lines):
            block_end = get_block_end(lines, i)
            block_lines = lines[i:block_end]
            block_text = '\n'.join(block_lines)
            
            if not block_text.strip():
                i = block_end
                continue
            
            if current_length + len(block_text) + 2 <= max_length:
                if current_message and block_text.strip():
                    current_message.append('')
                current_message.extend(block_lines)
                current_length += len(block_text) + 2
            else:
                if current_message:
                    messages.append('\n'.join(current_message).strip())
                    current_message = block_lines
                    current_length = len(block_text)
                else:
                    sub_parts = split_long_block(block_text, max_length)
                    if len(sub_parts) > 1:
                        messages.extend(sub_parts[:-1])
                    current_message = sub_parts[-1].split('\n')
                    current_length = len(sub_parts[-1])
            
            i = block_end
        
        if current_message:
            messages.append('\n'.join(current_message).strip())
        
        return [msg.strip() for msg in messages if msg.strip()]

    def split_long_block(text: str, max_length: int) -> List[str]:
        parts = []
        current_part = []
        current_length = 0
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[^a-zа-я])', text, flags=re.IGNORECASE)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) + 2 <= max_length:
                current_part.append(sentence)
                current_length += len(sentence) + 2
            else:
                if current_part:
                    parts.append(' '.join(current_part))
                
                if len(sentence) > max_length:
                    sub_sentences = re.split(r',\s+', sentence)
                    current_part = []
                    current_length = 0
                    
                    for sub in sub_sentences:
                        if current_length + len(sub) + 2 <= max_length:
                            current_part.append(sub)
                            current_length += len(sub) + 2
                        else:
                            if current_part:
                                parts.append(' '.join(current_part))
                            current_part = [sub]
                            current_length = len(sub)
                else:
                    current_part = [sentence]
                    current_length = len(sentence)
        
        if current_part:
            parts.append(' '.join(current_part))
        
        return parts

    async def safe_edit_message(bot: Bot, chat_id: int, message_id: int, text: str, 
                              parse_mode: Optional[str] = None, max_retries: int = 3) -> Optional[telegram.Message]:
        for attempt in range(max_retries):
            try:
                return await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=text,
                    parse_mode=parse_mode
                )
            except telegram.error.RetryAfter as e:
                if attempt < max_retries - 1:
                    retry_after = int(str(e).split()[-2])
                    await asyncio.sleep(retry_after + 1)
                    continue
                else:
                    raise
            except telegram.error.BadRequest as e:
                if "Message is not modified" not in str(e):
                    if "Message_too_long" in str(e):
                        return None
                    raise
        return None

    async def safe_send_message(bot: Bot, chat_id: int, text: str, 
                              parse_mode: Optional[str] = None, max_retries: int = 3) -> Optional[telegram.Message]:
        for attempt in range(max_retries):
            try:
                return await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode
                )
            except telegram.error.RetryAfter as e:
                if attempt < max_retries - 1:
                    retry_after = int(str(e).split()[-2])
                    await asyncio.sleep(retry_after + 1)
                    continue
                else:
                    raise
        return None
    
    async def gradually_edit_message(bot: Bot, chat_id: int, message_id: int, text: str, delay: float):
            """
            Отправляет сообщение с постепенным выводом текста, разбивая текст по предложениям, нумерованным блокам и заголовкам.
            """
            # Шаблон для поиска нумерованных списков, заголовков и конца предложений
            pattern = r'(\d+\.\s|\d+\)\s|(?<=[.!?])\s+|^#+\s)'  # Ищем конец предложения, начало нумерации или заголовок
            
            # Разбиваем текст с использованием шаблона
            parts = re.split(pattern, text, flags=re.MULTILINE)

            current_text = ""

            for i in range(0, len(parts), 2):
                # Собираем предложение, нумерованный блок или заголовок
                sentence = parts[i]
                if i + 1 < len(parts):
                    sentence += parts[i + 1]

                current_text += sentence + ' '  # Добавляем предложение и пробел
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)  # Печатает...
                await safe_edit_message(bot, chat_id, message_id, current_text.strip())
                await asyncio.sleep(delay)
    async def send_messages_with_typing_effect(bot: Bot, chat_id: int, text: str, delay: float):
        messages = smart_text_split(text)
        
        for i, message_part in enumerate(messages):
            if i == 0:
                # Для первого сообщения используем редактирование
                message = await safe_send_message(bot, chat_id, "...")
                await gradually_edit_message(bot, chat_id, message.message_id, message_part, delay)
            else:
                # Для последующих сообщений используем постепенную отправку
                await gradually_send_message(bot, chat_id, message_part, delay)
            
            # Добавляем небольшую паузу между сообщениями
            await asyncio.sleep(1)

    async def gradually_send_message(bot: Bot, chat_id: int, text: str, delay: float):
        pattern = r'(\d+\.\s|\d+\)\s|(?<=[.!?])\s+|^#+\s)'
        parts = re.split(pattern, text, flags=re.MULTILINE)

        current_text = ""
        message = await safe_send_message(bot, chat_id, "...")

        for i in range(0, len(parts), 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]

            current_text += sentence + ' '
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await safe_edit_message(bot, chat_id, message.message_id, current_text.strip())
            await asyncio.sleep(delay)
    
    
    async def gradually_send_message(bot: Bot, chat_id: int, text: str, delay: float):
            """
            Отправляет сообщение с постепенным выводом текста, разбивая текст по предложениям, нумерованным блокам и заголовкам.
            """
            # Шаблон для поиска нумерованных списков, заголовков и конца предложений
            pattern = r'(\d+\.\s|\d+\)\s|(?<=[.!?])\s+|^#+\s)'  # Ищем конец предложения, начало нумерации или заголовок
            
            # Разбиваем текст с использованием шаблона
            parts = re.split(pattern, text, flags=re.MULTILINE)

            current_text = ""

            for i in range(0, len(parts), 2):
                # Собираем предложение, нумерованный блок или заголовок
                sentence = parts[i]
                if i + 1 < len(parts):
                    sentence += parts[i + 1]

                current_text += sentence + ' '  # Добавляем предложение и пробел
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)  # Печатает...
                # await safe_edit_message(bot, chat_id, message_id, current_text.strip())
                await bot.send_message(chat_id, current_text.strip(), disable_notification=True)
                await asyncio.sleep(delay)
             
    # Разбиваем текст на части и отправляем
    messages = smart_text_split(text)
    
    # Первое сообщение редактируем с эффектом набора текста
    if messages:
        await gradually_edit_message(bot, chat_id, message_id, messages[0], delay)
    
    # Оставшиеся части отправляем новыми сообщениями
    for message_part in messages[1:]:
        await send_messages_with_typing_effect(bot, chat_id, message_part, delay=0.5)
        await asyncio.sleep(delay)


#################################################################################################################                 
# Эта функция - делит длинное сообщение на части
async def send_long_message(chat_id, message):
    # Разбиваем сообщение на части по 4096 символов
    for i in range(0, len(message), 4096):
        await bot.send_message(chat_id, message[i:i + 4096])
        # Добавим небольшую задержку, если необходимо
        await asyncio.sleep(0.1)

#################################################################################################################                 
# Эта функция - ответ на команду /start
async def start(update, context):
    await update.message.reply_text(f'''Добро пожаловать! Это бот-репетитор по английскому c gpt.''')
    context.user_data[key_name] = update.message.from_user.first_name


#################################################################################################################                 
# Эта функция - ответ на команду /help 
async def help_command(update, context):

    await update.message.reply_html(rf"Hi {user.mention_html()}! Это бот-репетитор по английскому c gpt.")

#################################################################################################################   
# Функция-обработчик текстовых сообщений:
async def handle_message(update, context):

    first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...', reply_to_message_id=update.message.message_id)
    user_message = update.message.text
    resp = await run_graph(user_message, update.message.chat_id)

    if resp:   
        #await update.message.reply_text(resp)
               
        await send_message_gradually(context.bot, update.message.chat_id, resp, first_message.message_id)
    else:
        await update.message.reply_text('gpt не ответил.')    

#################################################################################################################   
# # Функция-обработчик сообщений с голосовым сообщением (распознаем с виспером, генерируем с googleTTS):    
# async def gpt_v(update, context):   
    
#     first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...', reply_to_message_id=update.message.message_id)    
#     chat_id = str(update.message.chat_id)  # Получаем chat_id пользователя
#     user_dir = os.path.join("user_data", chat_id)  # Создаем директорию для пользователя, если она не существует
#     os.makedirs(user_dir, exist_ok=True)
#     file = await update.message.voice.get_file()
#     voice_as_byte = await file.download_as_bytearray()
#     byte_voice = BytesIO(voice_as_byte)  

#     audio = AudioSegment.from_file(byte_voice, format='ogg')  # Создаем объект AudioSegment из массива байт   
#     audio_path = os.path.join(user_dir, 'voice_message.mp3')  # Путь для сохранения MP3 файла   
#     audio.export(audio_path, format='mp3')          # Экспортируем аудиофайл в формат mp3

#     with open(audio_path, "rb") as audio_file:
#         transcription = await AsyncOpenAI().audio.transcriptions.create(
#             model="whisper-1", 
#             file=audio_file,
#             response_format="text"
#         )
    
#     resp = await run_graph(transcription, update.message.chat_id)

#     if resp:   
#         # бесплатный google TTS 
#         tts = gTTS(resp) #генерируем голосовой файл по тексту ответа gpt (переменная res) с помощью gtts 
   
#         output_path = os.path.join(user_dir, "output.mp3")      # Путь для сохранения выходного MP3 файла       
#         tts.save(output_path)       # Сохраняем аудиофайл в формате MP3
        
#         # Открываем аудиофайл для отправки пользователю
#         with open(output_path, 'rb') as audio_file:
#             await update.message.reply_voice(audio_file)
#         #await update.message.reply_text(resp)  
#         await send_message_gradually(context.bot, update.message.chat_id, resp, first_message.message_id)
#     else:
#         await update.message.reply_text('gpt не ответил.')    

#################################################################################################################      
# Основная функция для запуска бота
def main():

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start, block=False))
    application.add_handler(CommandHandler("help", help_command, block=False))

    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.REPLY, handle_message, block=False))
    #application.add_handler(MessageHandler(filters.TEXT & filters.REPLY, handle_confirmation, block=False))
    # application.add_handler(MessageHandler(filters.VOICE, gpt_v, block=False))

    
    # запуск приложения. Для остановки нужно нажать Ctrl-C
    # Запускаем асинхронную очередь
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
