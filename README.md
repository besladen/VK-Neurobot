# vk-markovify-chatbot

Бот, генерирующий сообщения Марковским процессом на основе сообщений из беседы. Для каждой беседы ведёт отдельную историю сообщений в txt.

## Установка

- Устанавливаем [Python](https://python.org/downloads) (для Windows 7 нужен Python 3.8.X). Во время установки обязательно ставим галочку `Add Python to PATH (Добавить Python в PATH)`.
- [Скачиваем архив с ботом](https://github.com/monosans/vk-markovify-chatbot/archive/refs/heads/main.zip).
- Распаковываем архив.
- Переходим в группу, в которой размещаем бота:
  1. Управление
  2. Настройки
  3. Работа с API
  4. Создать ключ
  5. Выставляем галочки и создаем
  6. Копируем и вставляем полученный токен в `config.ini`
  7. При желании настраиваем прочие параметры в `config.ini`
- Настраиваем Long Poll API:
  1. Управление
  2. Настройки
  3. Работа с API
  4. Вкладка Long Poll API
  5. Long Poll API: Включено + Версия API: самая новая
  6. Вкладка Типы событий
  7. Ставим все галочки раздела "Сообщения"
- Даём группе возможность писать сообщения и позволяем добавлять её в беседы:
  1. Управление
  2. Сообщения
  3. Сообщения сообщества: Включены
  4. Настройки для бота
  5. Возможности ботов: Включены
  6. Разрешать добавлять сообщество в беседы - ставим галочку

## Запуск

- `bot.py` - понадобится самостоятельно установить библиотеки из `requirements.txt`.
- или `start.cmd` на Windows и `start.sh` на \*nix - нужные библиотеки установятся автоматически.

## License / Лицензия

[MIT](LICENSE)
