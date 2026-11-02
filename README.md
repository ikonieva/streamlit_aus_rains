# Streamlit Australian Rains

Цей проєкт демонструє розгортання моделі машинного навчання за допомогою Streamlit для прогнозування дощу в Австралії. Додаток дозволяє користувачам вводити характеристики поточної погоди та отримувати прогноз на основі навченої моделі Logistic Regression Forest.

Протестувати роботу аплікейшену можна за посиланням: 
Якщо за посиланням виводиться повідомлення "This app has gone to sleep due to inactivity. Would you like to wake it back up?" просто натисніть кнопку "Yes, get this app back up!" і зачекайте півхвилини.

## Структура проєкту

- **data/**: Директорія, що містить набір даних (`weatherAUS.csv`).
- **images/**: Директорія для зберігання зображень, які використовуються в додатку.
- **model/**: Директорія, що містить навчену ML-модель.
- **app.py**: Основний файл застосунку Streamlit.
- **requirements.txt**: Список необхідних Python-пакетів.

## Налаштування

### Передумови

Переконайтеся, що у вас встановлений Python 3.12 або новішої версії. Також потрібно встановити необхідні пакети, зазначені в `requirements.txt`.

### Встановлення

1. **Клонуйте репозиторій**:
   ```bash
   git clone https://github.com/your-username/streamlit_aus_rains.git
   cd streamlit_aus_rains
   ```

2. **Створіть віртуальне середовище** (необов'язково, але рекомендовано):
   ```bash
   python -m venv venv
   source venv/bin/activate   # На Windows використовуйте `venv\Scripts\activate`
   ```

3. **Встановіть необхідні пакети**:
   ```bash
   pip install -r requirements.txt
   ```

### Запуск додатку Streamlit

Запустіть додаток Streamlit локально за допомогою наступної команди:

```bash
streamlit run app.py
```

Додаток буде доступний за адресою `http://localhost:8501`.
