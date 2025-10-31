import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from io import BytesIO


# ===================================================================================
# Классическая модель
# ===================================================================================

class ClassicModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_model(self, X_train, y_train):
        """Обучаем модель"""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Прогнозируем с использованием модели"""
        return self.model.predict(X)

    def retrain_and_predict_next_30_days(self, feature, target, days_for_predict=30):
        """Переобучаем модель на всех данных"""
        self.model.fit(feature, target)
        """Прогнозируем на 30 дней вперёд после переобучения модели на всех данных"""
        X_last = feature.iloc[-1:].copy()

        predict_next = []
        for _ in range(days_for_predict):

            pred = self.model.predict(X_last)  # Прогнозируем следующую цену
            predict_next.append(pred[0].round(4))
            # Обновляем входные данные для следующего шага
            X_last = X_last.shift(1, axis=1)  # Сдвиг данных
            X_last[["lag_1"]] = pred[0]  # Добавляем прогноз в лаг 1

        return predict_next

# ===================================================================================
# Статическая модель
# ===================================================================================


class SarimaModel:
    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def train_model(self, train):
        # Убедимся, что индекс с частотой
        train = train.asfreq('B').interpolate()

        self.model = SARIMAX(
            train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=500)

    def predict(self, steps):
        forecast = self.model.get_forecast(steps)
        return forecast.predicted_mean

    def retrain_and_predict_next_30_days(self, feature, target, days_for_predict=30):
        """
        Переобучаем SARIMA на всех данных 
        и прогнозируем на n дней вперед.
        """
        # Объединяем Series, убеждаемся, что индекс с частотой
        target_full = target.asfreq('B').interpolate()

        # Переобучаем модель на всех данных
        self.model = SARIMAX(
            target_full,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=1000)

        # Прогнозируем на следующие n дней
        forecast = self.model.get_forecast(days_for_predict)

        return list(round(forecast.predicted_mean, 4))


# ===================================================================================
# Нейросетевая модель
# ===================================================================================
class LSTMModel:
    def __init__(self, lag, epochs=60, batch_size=16, learning_rate=0.001):
        self.lag = lag
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None

    def _reshape_for_lstm(self, X):
        """LSTM ожидает вход: (samples, timesteps, features)"""
        return np.expand_dims(X, axis=2)

    def build_model(self):
        """Создаём архитектуру LSTM"""
        model = Sequential([
            Input(shape=(self.lag, 1)),
            LSTM(64, activation='tanh'),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(
            learning_rate=self.learning_rate), loss='mse')
        return model

    def train_model(self, X_train, y_train):
        """Обучаем модель на тренировочных данных"""
        X_train_lstm = self._reshape_for_lstm(X_train)
        self.model = self.build_model()
        es = EarlyStopping(monitor='loss', patience=5,
                           restore_best_weights=True)
        self.model.fit(
            X_train_lstm, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[es]
        )

    def predict(self, X):
        """Делаем прогноз на тестовых данных"""
        X_lstm = self._reshape_for_lstm(X)
        return self.model.predict(X_lstm, verbose=0).flatten()

    def retrain_and_predict_next_30_days(self, feature, target, days_for_predict=30):
        """Переобучаем модель на всех данных и прогнозируем на n дней вперед"""

        # Преобразуем в float32 numpy
        X_all_lstm = self._reshape_for_lstm(feature.values.astype(np.float32))
        y_all = target.values.astype(np.float32)

        # Переобучаем модель
        self.model.fit(
            X_all_lstm, y_all,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )

        # Пошаговый прогноз
        preds = []
        X_last = feature.iloc[-1:].copy()

        for _ in range(days_for_predict):
            X_input = self._reshape_for_lstm(X_last.values.astype(np.float32))
            pred = self.model.predict(X_input, verbose=0)[0, 0]
            preds.append(np.round(pred, 4))

            # Обновляем лаги
            X_last = X_last.shift(1, axis=1)
            X_last.iloc[0, 0] = pred  # обновляем lag_1

        return preds


# ===================================================================================
# Пайплайн для запуска
# ===================================================================================
class TimeSeriesPipeline:
    '''
    Общий пайплайн для 3 моделей
    Запускает 3 модели, сравнивает метрики, выбирает лучшую модель, для нее достраивает график на следующие 30 дней.
    '''

    def __init__(self, data, user_ticker, lag=10, days_for_predict=30, test_size=0.2):
        self.data = data
        self.lag = lag
        self.user_ticker = user_ticker
        self.days_for_predict = days_for_predict
        self.test_size = test_size
        self.metrics = pd.DataFrame(columns=["Model", "mae", "rmse", "mape"])
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.target = None
        self.results = None

        self.create_features()

    def create_features(self):
        """Создаём лаги для временного ряда и делим данные на train/test"""
        df = self.data.copy()
        for i in range(1, self.lag + 1):
            df[f'lag_{i}'] = df['price'].shift(i)

        # Убираем пропуски, которые возникли из-за сдвигов
        df = df.dropna()

        self.features = df.drop(columns=['price'])
        self.target = df['price']

        # Разделение на train/test (данные всегда разбиваются 80/20 без перемешивания)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=self.test_size, shuffle=False)

    def evaluate(self, y_true, y_pred):
        "Рассчет метрик на всех моделях"
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) /
                       np.maximum(y_true, 1e-8))) * 100
        return np.round(mae, 4), np.round(rmse, 4), np.round(mape, 4)

#  -------------------------Запуск RandomForest------------------------------
    def run_classic(self):

        cls_model = ClassicModel()  # Создаём экземпляр модели
        self.models['RandomForest'] = cls_model

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        # Обучение модели
        cls_model.train_model(X_train, y_train)

        # Прогнозирование на тестовых данных
        y_pred = cls_model.predict(X_test)

        # Оценка модели
        mae, rmse, mape = self.evaluate(y_test, y_pred)

        # добавляем в общий датасет с метриками
        self.metrics.loc[len(self.metrics)] = ['RandomForest', mae, rmse, mape]

#  -------------------------Запуск Sarima------------------------------

    def run_sarima(self):
        sarima = SarimaModel()
        self.models['SARIMA'] = sarima

        train, test = self.y_train, self.y_test

        sarima.train_model(train)
        pred = sarima.predict(len(test))

        mae, rmse, mape = self.evaluate(test, pred)
        self.metrics.loc[len(self.metrics)] = ['SARIMA', mae, rmse, mape]

#  -------------------------Запуск LSTM ------------------------------
    def run_lstm(self):
        """Запуск LSTM модели"""
        lstm = LSTMModel(self.lag)
        self.models['LSTM'] = lstm
        lstm.train_model(self.X_train.values, self.y_train.values)
        y_pred = lstm.predict(self.X_test.values)
        mae, rmse, mape = self.evaluate(self.y_test, y_pred)
        self.metrics.loc[len(self.metrics)] = ['LSTM', mae, rmse, mape]

#  -------------------------Выбор лучшей модели------------------------------

    def choice_model(self, mae_threshold=0.05, rmse_threshold=0.05):

        # минимальные MAE и RMSE
        min_mae = self.metrics['mae'].min()
        min_rmse = self.metrics['rmse'].min()

        #  Отобрать модели, близкие к лучшим по MAE и RMSE, если они близки
        candidates = self.metrics[
            (self.metrics['mae'] <= min_mae * (1 + mae_threshold)) &
            (self.metrics['rmse'] <= min_rmse * (1 + rmse_threshold))
        ]

        # 3. Среди кандидатов выбираем минимальный MAPE
        best_model_row = candidates.loc[candidates['mape'].idxmin()]

        return best_model_row


#  -------------------------Прогноз на 30 дней------------------------------


    def month_predict(self,  best_model_name):

        best_model = self.models[best_model_name]

        self.results = best_model.retrain_and_predict_next_30_days(
            self.features, self.target)

        return self.results


#  -------------------------Построение графика------------------------------

    def plot_forecast(self, days=None):

        if not days:
            data = self.data
        else:
            data = self.data[-days:]
        last_date = data.index.max()
        last_value = data['price'].iloc[-1]

        # Создаем даты для прогноза
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=len(self.results))

        # Добавляем последнюю фактическую точку перед прогнозом
        forecast_df = pd.DataFrame(
            self.results, index=forecast_dates, columns=['price'])
        forecast_df.loc[last_date] = last_value
        forecast_df = forecast_df.sort_index()

        # Строим график
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['price'],
                 label='Исторические данные', color='blue')
        plt.plot(forecast_df.index,
                 forecast_df['price'], label='Прогноз', color='red')
        plt.axvline(x=last_date, color='gray',
                    linestyle=':', label='Начало прогноза')
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.title(f'Исторические данные и прогноз для {self.user_ticker}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Сохраняем в память
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf


#  -------------------------Составление рекомендаций------------------------------

    def get_simple_recommendations(self, user_money=10):

        current_price = self.data['price'].iloc[-1]
        forecast_prices = self.results
        end_of_period_price = forecast_prices[-1]

        # Создаем даты для прогноза
        last_date = self.data.index.max()
        forecast_dates = [
            last_date + pd.Timedelta(days=i+1) for i in range(len(forecast_prices))]

        # Анализируем прогноз
        avg_forecast = np.mean(forecast_prices)
        return_percent = ((avg_forecast - current_price) / current_price) * 100

        # Пассивная стратегия
        if return_percent > 10:
            passive_action = "✅ ПОКУПАТЬ"
            passive_profit = user_money * return_percent / 100
        elif return_percent > 0:
            passive_action = "⚖️ ДЕРЖАТЬ"
            passive_profit = user_money * return_percent / 100
        else:
            passive_action = "❌ НЕ ПОКУПАТЬ"
            passive_profit = 0

        # Агрессивная стратегия - находим точки сделок
        buy_points = []
        sell_points = []

        # Поиск точек покупки/продажи
        for i in range(1, len(forecast_prices)-1):
            if forecast_prices[i] < forecast_prices[i-1] and forecast_prices[i] < forecast_prices[i+1]:
                buy_points.append(i)
            elif forecast_prices[i] > forecast_prices[i-1] and forecast_prices[i] > forecast_prices[i+1]:
                sell_points.append(i)

        # Формируем список сделок
        trades_list = []
        for i in buy_points:
            trades_list.append({
                'date': forecast_dates[i],
                'action': 'КУПИТЬ',
                'price': forecast_prices[i],
                'day': i+1
            })

        for i in sell_points:
            trades_list.append({
                'date': forecast_dates[i],
                'action': 'ПРОДАТЬ',
                'price': forecast_prices[i],
                'day': i+1
            })

        # Сортируем сделки по дате
        trades_list.sort(key=lambda x: x['day'])

        # Форматируем таблицу сделок
        trades_table = "━━━━━━━━━━━━━━━━\n"
        if trades_list:
            for trade in trades_list:
                date_str = trade['date'].strftime('%d.%m')
                action = "📈 КУПИТЬ" if trade['action'] == 'КУПИТЬ' else "📉 ПРОДАТЬ"
                trades_table += f"{date_str} | {action} | ${trade['price']:.2f}\n"

                # Расчет прибыли для агрессивной стратегии
        aggressive_profit = 0
        if buy_points and sell_points:
            paired_trades = []
            for b in buy_points:
                # Находим ближайшую продажу после покупки
                next_sell = [s for s in sell_points if s > b]
                if next_sell:
                    s = next_sell[0]
                    profit = user_money * \
                        (forecast_prices[s] / forecast_prices[b] - 1)
                    aggressive_profit += profit
                    paired_trades.append((b, s, profit))

            if aggressive_profit <= 0:
                trades_table = "⚠️ Агрессивная стратегия убыточна на данном промежутке времени\n"

            # Формируем сообщение
        message = f"""
            💰 *Рекомендации для инвестиций ${user_money:,.0f}*

            ━━━━━━━━━━━━━━━━
            📈 *Пассивная стратегия:*
            {passive_action}
            • Текущая цена: ${current_price:.2f}
            • Ожидаемая цена в конце периода: ${end_of_period_price:.2f}
            • Разница между текущей и ожидаемой ценой : ${round((end_of_period_price - current_price), 3)}
            • Средняя прогнозная: ${avg_forecast:.2f}
            • Ожидаемая доходность: {return_percent:.1f}%
            • Прибыль: ${passive_profit:.1f}
            ━━━━━━━━━━━━━━━━
            🎯 *Агрессивная стратегия:*
            • Сигналов: 📈 {len(buy_points)} | 📉 {len(sell_points)}
            • Ожидаемая прибыль: ${aggressive_profit:.1f}

            📊 *Ближайшие сделки:*\n
            {trades_table}
            ━━━━━━━━━━━━━━━━
           
            💡*Disclaimer* Бот создан в учебных целях.
              *Не является финансовой рекомендацией!*
            """

        return message, passive_profit, aggressive_profit
