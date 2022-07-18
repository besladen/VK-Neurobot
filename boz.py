#!/usr/bin/env python3
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "Kirili4ik/ruDialoGpt3-medium-finetuned-telegram"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
model.eval()
import asyncio
import logging
import random
import re
from configparser import ConfigParser

import markovify
from aiofiles import open as aopen
from aiofiles import os as aos
from pydantic import BaseModel, Field, NonNegativeFloat
from vkbottle import VKAPIError
from vkbottle.bot import Bot, Message
from vkbottle.dispatch.rules.base import ChatActionRule, FromUserRule
from vkbottle_types.objects import MessagesMessageActionStatus
from vkbottle.tools import DocMessagesUploader

class Config(BaseModel):
    bot_token: str = Field(min_length=1)
    response_delay: NonNegativeFloat
    response_chance: float = Field(gt=0, le=100)

    class Config:
        anystr_strip_whitespace = True
        validate_assignment = True


def get_config() -> Config:
    config = ConfigParser()
    config.read("config.ini", encoding="utf-8")
    cfg = config["DEFAULT"]
    return Config(
        bot_token=cfg.get("BotToken"),
        response_delay=cfg.getfloat("ResponseDelay", 0),
        response_chance=cfg.getfloat("ResponseChance", 100),
    )


config = get_config()
bot = Bot(config.bot_token)
bot.loop_wrapper.on_startup.append(aos.makedirs("db", exist_ok=True))
tag_pattern = re.compile(r"\[(id\d+?)\|.+?\]")
empty_line_pattern = re.compile(r"^\s+", flags=re.M)


@bot.on.message(text="🐱")
async def hi_handler(message: Message):
    await message.answer("Наташа ты покормила кота?")
  
@bot.on.message(text="//Да")
async def hiz_handler(message: Message):
    await message.answer("Я Даун")

@bot.on.message(text="/База")
async def hii_handler(message: Message):
    a = ["video-206886750_456240015", "video439524667_456240725", "video514749102_456239196"]
    await message.answer(attachment=random.choice(a))

@bot.on.message(text="/татыч")
async def hiw_handler(message: Message):
    await message.answer("Татыч ты секси")

@bot.on.message(text="/Виталя")
async def hiq_handler(message: Message):
    await message.answer("Мне кажется ты мой брат, мы с тобой явно из одной двачерской семьи")

@bot.on.message(text="🚑")
async def hix_handler(message: Message):
    await message.answer("У меня инсульт")

@bot.on.message(text="/Да")
async def hic_handler(message: Message):
    await message.answer("Пизда")    

@bot.on.chat_message(text=["/@$#", "/@$#"])  # type: ignore[misc]
async def reset(message: Message) -> None:
    """Сброс базы данных администратором беседы."""
    try:
        members = await message.ctx_api.messages.get_conversation_members(
            peer_id=message.peer_id
        )
    except VKAPIError[917]:
        await message.reply(
            "Не удалось проверить, являетесь ли вы администратором,"
            + " потому что я не администратор."
        )
        return
    admins = {member.member_id for member in members.items if member.is_admin}
    if message.from_id in admins:

        # Удаление базы данных беседы
        try:
            await aos.remove(f"db/{message.peer_id}.txt")
        except FileNotFoundError:
            pass

        reply = f"@id{message.from_id}, база данных успешно сброшена."
    else:
        reply = "Сбрасывать базу данных могут только администраторы."
    await message.reply(reply)
def get_length_param(text: str, tokenizer) -> str:
    tokens_count = len(tokenizer.encode(text))
    if tokens_count <= 15:
        len_param = '1'
    elif tokens_count <= 50:
        len_param = '2'
    elif tokens_count <= 256:
        len_param = '3'
    else:
        len_param = '-'
    return len_param




@bot.on.chat_message(FromUserRule())  # type: ignore[misc]
async def talk(message: Message) -> None:
    text = message.text.lower()
    file_name = f"db/{message.peer_id}.txt"
    input_user = (text)
    if text:
        # Удаление пустых строк
        text = empty_line_pattern.sub("", text)

        # Преобразование [id1|@durov] в @id1
        text = tag_pattern.sub(r"@\1", text)

        # Запись сообщения в историю беседы
        async with aopen(file_name, "a", encoding="utf-8") as f:
            await f.write(f"\n{text}")
    elif not await aos.path.exists(file_name):
        return

    if random.random() * 100 > config.response_chance:
        return

    # Задержка перед ответом
    await asyncio.sleep(config.response_delay)

    # Чтение истории беседы
    async with aopen(file_name, encoding="utf-8") as f:
        db = await f.read()
    db = db.strip().lower()

    # Генерация сообщения
    text_model = markovify.NewlineText(
        input_text=db, state_size=2, well_formed=False
    )
    sentence = text_model.make_sentence(max_overlap_ratio=0.3,tries=10000) or random.choice(
        db.splitlines()
    )



    chat_history_ids = torch.zeros((1, 0), dtype=torch.int)



# encode the new user input, add parameters and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(f"|0|{get_length_param(input_user, tokenizer)}|" \
                                              + input_user + tokenizer.eos_token, return_tensors="pt")
# append the new user input tokens to the chat history
    chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    next_len = "3"  # input("Exp. len?(-/1/2/3): ")
# encode the new user input, add parameters and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(f"|1|{next_len}|", return_tensors="pt")
# append the new user input tokens to the chat history
    chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

# print(tokenizer.decode(chat_history_ids[-1])) # uncomment to see full gpt input

# save previous len
    input_len = chat_history_ids.shape[-1]
# generated a response; PS you can read about the parameters at hf.co/blog/how-to-generate
    chat_history_ids = model.generate(
            chat_history_ids,
            num_return_sequences=1,  # use for more variants, but have to print [i]
            max_length=100,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=1.0,  # 0 for greedy
            mask_token_id=tokenizer.mask_token_id,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            device='cpu'
        )





        # pretty print last ouput tokens from bot
    bototvet = (tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True))
    await message.answer(bototvet)

    

@bot.on.chat_message(mention=True)
async def mention_handler(message: Message):
    await message.reply("Привет, чего вы хотите?")    

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        import uvloop
    except ImportError:
        pass
    else:
        uvloop.install()

    bot.run_forever()


if __name__ == "__main__":
    main()

