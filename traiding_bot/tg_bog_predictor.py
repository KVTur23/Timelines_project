import os
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.exceptions import TelegramNetworkError
from predict_models import TimeSeriesPipeline
from aiogram.types import BufferedInputFile


#  ==========================НАСТРОЙКА ОКРУЖЕНИЯ ==========================

load_dotenv()  # Загружаем переменные из .env

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Не найден TELEGRAM_BOT_TOKEN в .env")


# Загружаем данные NASDAQ
url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed-symbols.csv"
tickers_name = pd.read_csv(url)["Symbol"].tolist()
tickers = pd.read_csv(url)[["Symbol", "Company Name"]].dropna()

# Директории для хранения логов
LOGS_DIR = Path("log_files")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "user_logs.csv"
# Словарь для хранения состояния пользователя
user_state = {}

#  ==========================ИНИЦИАЛИЗАЦИЯ БОТА ==========================

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


# ========================== ЛОГИРОВАНИЕ ==========================

def log_user_action(
    user_id,
    query_time,
    query_text,
    ticker,
    company_current_price,
    money,
    best_model,
    mae,
    rmse,
    mape,
    passive_profit,
    aggressive_profit
):
    """
     Логирует действия пользователя в CSV-файл.
     Сохраняет все три метрики (MAE, RMSE, MAPE), рассчитанную прибыль
     и полный текст запроса пользователя.
     Отсутствующие значения безопасно записываются как NaN.
     """

    def safe_round(value, ndigits=4):
        return round(value, ndigits) if isinstance(value, (int, float)) else np.nan

    log_df = pd.DataFrame([{
        "user_id": user_id,
        "datetime": query_time.strftime("%Y-%m-%d %H:%M:%S"),
        "query_text": query_text or "N/A",
        "ticker": ticker or "N/A",
        "company_current_price": company_current_price or "N/A",
        "money": money,
        "best_model": best_model or "N/A",
        "MAE": safe_round(mae, 4),
        "RMSE": safe_round(rmse, 4),
        "MAPE": safe_round(mape, 2),
        "passive_profit_estimate": safe_round(passive_profit, 2),
        "aggressive_profit_estimate": safe_round(aggressive_profit, 2),

    }])

    # Проверяем, существует ли файл
    if LOG_FILE.exists():
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_FILE, mode="w", header=True, index=False)


#  ==========================ОБРАБОТЧИКИ КОМАНД ==========================

@dp.message(Command("start"))
async def start_command(message: Message):
    """Приветственное сообщение при старте."""
    user_id = message.from_user.id
    welcome_text = (
        "🌟 Приветствую! Я бот для анализа и прогнозирования акций.\n\n"
        "💹 Для старта отправь тикер компании сумму инвестиций через пробел.\n"
        "🔍 Могу помочь найти тикер, если не знаешь — просто напиши название компании .\n"
        "📊 Я подготовлю для тебя краткий финансовый обзор и прогноз 📉📈.\n\n"
        "🚀 Время увеличивать свой капитал 💎💸!"
    )
    await message.answer(welcome_text)


@dp.message(F.text)
async def process_message(message: Message):
    """Обработка пользовательского ввода: тикер + сумма.
    Запуск пайплайна из 3 моделей, отправка результато пользователю.
    """
  #  ---------------------Обработка пользовательского сообщения  ---------------------
    user_id = message.from_user.id
    query_time = datetime.now()
    query_text = message.text.strip()

    # Инициализация состояния пользователя
    if user_id not in user_state:
        user_state[user_id] = {"money": 0, "ticker": None}

     # Инициализация переменных
    user_money = user_state[user_id]["money"]
    user_ticker = None
    best_model_name = None
    mae = None
    rmse = None
    mape = None
    passive_profit = None
    aggressive_profit = None
    company_current_price = None

    # Разбираем сообщение
    if query_text.split(" ")[-1].isnumeric():
        user_money = int(query_text.split(" ")[-1])
        user_state[user_id]["money"] = user_money
        user_company_list = query_text.split(" ")[:-1]
    else:
        user_company_list = query_text.split(" ")
        if user_money == None:
            user_money = 0
            await message.answer("Кажется, вы забыли указать сумму, в таком случае будет предоставлен прогноз без рекомендаций.")

    # Пустой датафрейм для накопления результатов
    all_results = pd.DataFrame(columns=tickers.columns)

    # Проверяем, есть ли тикер среди введенного
    for query in user_company_list:
        if query.upper() in tickers_name:
            user_ticker = query.upper()
            user_state[user_id]["ticker"] = user_ticker
            await message.answer(f"Выбранная компания: {user_ticker}, сумма: {user_money}")

    # Если тикер не найден, ищем по подстроке в названиях компаний предлагаем подходящие
    if user_ticker is None:
        for query in user_company_list:
            filtered = tickers[tickers['Company Name'].str.contains(
                query, case=False, na=False)]
            if not filtered.empty:
                all_results = pd.concat(
                    [all_results, filtered], ignore_index=True)

        if len(all_results) > 20:
            all_results = all_results.head(20)
        all_results = all_results.rename(
            columns={'Symbol': 'Тикер', 'Company Name': 'Компания'})
        if all_results.empty:
            await message.reply("Не найдено подходящих тикеров, повторите запрос.")
        else:
            await message.answer(f"Список подходящих тикеров:\n{'━'*15}\n{all_results.to_string(index=False)}\n{'━'*15}\nВыберите нужный тикер или введите запрос заново и при необходимости укажите сумму через пробел")

    else:
        try:
            info = yf.Ticker(user_ticker).info
            user_id = message.from_user.id

            # Отправляем краткую информацию пользователю
            company_name = info.get(
                "longName", info.get("displayName", user_ticker))
            company_current_price = info.get('regularMarketPrice', 'N/A')
            response = (
                f"*{company_name}*\n"
                f"{'━'*15}\n"
                f"🌍 *Страна:* {info.get('country', 'N/A')}\n"
                f"🏙️ *Город:* {info.get('city', 'N/A')}\n"
                f"🏢 *Сектор:* {info.get('sector', 'N/A')}\n"
                f"💸 *Рыночная капитализация:* {info.get('marketCap', 'N/A'):,}\n"
                f"💲 *Текущая цена:* {info.get('regularMarketPrice', 'N/A')}\n"
                f"\n\n Готовлю рекомендации, это может занять пару минут...\n"
            )
            await message.answer(response, parse_mode="Markdown")
        except Exception as e:
            await message.answer(f"Данного тикера не оказалось в базе yfinance попробуйте другой")
            return  # Завершаем функцию при ошибке

 #  ---------------------Запуск пайплайна из 3 моделей  ---------------------

        data = yf.download(user_ticker, period="2y", auto_adjust=True)["Close"]
        # Привел удобному формату для пайплайна
        tick_data = pd.DataFrame(
            {'Date': data.index, 'price': data[user_ticker].values}).set_index("Date")

        # Используем пайплайн
        pipeline = TimeSeriesPipeline(
            tick_data, user_ticker, lag=10, days_for_predict=30)
        pipeline.run_classic()
        pipeline.run_sarima()
        pipeline.run_lstm()

        best_model_row = pipeline.choice_model()

        mae = best_model_row['mae']
        rmse = best_model_row['rmse']
        mape = best_model_row['mape']

 #  ---------------------Отправляем отчет о выбранной модели ---------------------

        model_info = f"""
        🎯 <b>Лучшая модель прогнозирования</b>

        ━━━━━━━━━━━━━━━
        🤖 <b>Модель:</b> {best_model_row['Model']}
        📊 <b>MAE:</b> {mae:.4f}
        📈 <b>RMSE:</b> {rmse:.4f}
        📉 <b>MAPE:</b> {mape:.2f}%
        ━━━━━━━━━━━━━━━

        • MAE - Средняя абсолютная ошибка
        • RMSE - Среднеквадратичная ошибка  
        • MAPE - Средняя абсолютная процентная ошибка
        """

        await message.answer(model_info, parse_mode="HTML")

        # имя модели
        best_model_name = best_model_row['Model']
        pipeline.month_predict(best_model_name)

 #  ---------------------График за 2 года ---------------------
        forecast_image = pipeline.plot_forecast()  # BytesIO объект

        # Создаем BufferedInputFile
        photo_file = BufferedInputFile(
            forecast_image.getvalue(),
            filename="forecast.png"
        )

        await message.answer_photo(
            photo=photo_file,
            caption=f"Прогноз для {pipeline.user_ticker}"
        )

        # Закрываем BytesIO
        forecast_image.close()

#  ---------------------График за 2 месяца---------------------

        forecast_image = pipeline.plot_forecast(30)  # BytesIO объект

        # Создаем BufferedInputFile
        photo_file = BufferedInputFile(
            forecast_image.getvalue(),
            filename="forecast.png"
        )

        await message.answer_photo(
            photo=photo_file,
            caption=f"Прогноз для {pipeline.user_ticker} в масштабе"
        )

        # Закрываем BytesIO
        forecast_image.close()
        aggressive_profit, passive_profit = 0, 0

#  ---------------------Отправляем рекомендации пользователю---------------------
        if user_money != 0:
            recommendation, passive_profit, aggressive_profit = pipeline.get_simple_recommendations(
                user_money)
            await message.answer(recommendation, parse_mode="Markdown")

        # Обнуляем запрос пользователя
        user_state[user_id] = {"money": None, "ticker": None}

#  -------------------------Записываем лог------------------------------
    log_user_action(
        user_id=user_id,
        query_time=query_time,
        query_text=query_text,
        ticker=user_ticker,
        company_current_price=company_current_price,
        money=user_money,
        best_model=best_model_name,
        mae=mae,
        rmse=rmse,
        mape=mape,
        passive_profit=passive_profit,
        aggressive_profit=aggressive_profit,
    )


#  ==========================ЗАПУСК ПРИЛОЖЕНИЯ ==========================
async def main():
    """Основная точка входа."""
    print("🤖 Бот запущен и ожидает сообщений...")
    try:
        await dp.start_polling(bot)
    except TelegramNetworkError:
        print("⚠️ Проблема с подключением к Telegram. Перезапуск...")
        await asyncio.sleep(5)
        await main()


if __name__ == "__main__":
    asyncio.run(main())
