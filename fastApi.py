from enum import IntEnum, Enum
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import numpy as np
import sklearn
from fastapi import File, UploadFile, HTTPException
from enum import IntEnum, Enum
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import numpy as np
import sklearn
import pandas as pd
import io

from starlette.responses import StreamingResponse

app = FastAPI()

model = joblib.load('lr1.joblib')

class CarFeatures(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

def preprocess_features(features: CarFeatures) -> np.ndarray:
    mileage = features.mileage
    engine = features.engine
    max_power = features.max_power
    torque = features.torque
    max_torque_rpm = features.torque
    if ' km/kg' in mileage:
        mileage = float(mileage.replace(' km/kg', '')) * 0.5
    elif ' kmpl' in mileage:
        mileage = float(mileage.replace(' kmpl', ''))
    if ' CC' in engine:
        engine = float(engine.replace(' CC', ''))
    if ' bhp' in max_power:
        max_power = float(max_power.replace(' bhp', ''))

    torque, sep, max_torque_rpm = torque.partition("@")

    if ' Nm' in torque:
        torque = float(torque.replace(' Nm', ''))
    elif 'Nm' in torque:
        torque = float(torque.replace('Nm', ''))
    elif 'nm' in torque:
        torque = float(torque.replace('nm', ''))
    elif ' nm' in torque:
        torque = float(torque.replace(' nm', ''))
    elif 'kgm' in torque:
        torque = float(torque.replace('kgm', '')) * 9.8

    if 'rpm' in max_torque_rpm:
        text = max_torque_rpm
        pattern = r"(\d+)-(\d+)rpm"
        match = re.search(pattern, text)
        if match:
            low = int(match.group(1))  # Нижняя граница (1500)
            high = int(match.group(2)) # Верхняя граница (2500)
            max_torque_rpm = (low + high) / 2 # Среднее значение

    if 'rpm' in max_torque_rpm:
        text = max_torque_rpm
        pattern = r"(\d+)-(\d+)\s*rpm"
        match = re.search(pattern, text)
        if match:
          low = int(match.group(1))  # Нижняя граница (1500)
          high = int(match.group(2)) # Верхняя граница (2500)
          max_torque_rpm = (low + high) / 2 # Среднее значение

    if 'rpm' in max_torque_rpm:
        max_torque_rpm = float(max_torque_rpm.replace('rpm', ''))
    elif ' rpm' in max_torque_rpm:
        max_torque_rpm = float(max_torque_rpm.replace(' rpm', ''))
    # Преобразование входных данных в массив numpy
    processed_features = np.array(
        [[features.year, features.km_driven, mileage, engine, max_power, torque, features.seats,
          max_torque_rpm]])
    return processed_features

def DataPreprocessing(df_test):
    for i in range(1000):
        if isinstance(df_test.at[i, 'mileage'], float):
            pass
        elif ' km/kg' in df_test.at[i, 'mileage']:
            df_test.at[i, 'mileage'] = float(df_test.at[i, 'mileage'].replace(' km/kg', '')) * 0.5
        elif ' kmpl' in df_test.at[i, 'mileage']:
            df_test.at[i, 'mileage'] = float(df_test.at[i, 'mileage'].replace(' kmpl', ''))
    for i in range(1000):
        if isinstance(df_test.at[i, 'engine'], float):
            pass
        elif ' CC' in df_test.at[i, 'engine']:
            df_test.at[i, 'engine'] = float(df_test.at[i, 'engine'].replace(' CC', ''))
    for i in range(1000):
        if isinstance(df_test.at[i, 'max_power'], float):
            pass
        elif ' bhp' in df_test.at[i, 'max_power']:
            df_test.at[i, 'max_power'] = float(df_test.at[i, 'max_power'].replace(' bhp', ''))
    median_value = df_test['engine'].median()
    df_test['engine'] = df_test['engine'].fillna(median_value)
    median_value = df_test['mileage'].median()
    df_test['mileage'] = df_test['mileage'].fillna(median_value)
    median_value = df_test['max_power'].median()
    df_test['max_power'] = df_test['max_power'].fillna(median_value)
    df_test['torque_copy'] = df_test['torque'].copy()
    import re
    for i in range(1000):
        if isinstance(df_test.at[i, 'torque_copy'], str):
            pattern = r"([\d.]+)@\s*([\d,]+)\((\w+)@\s*(\w+)\)"
            text = df_test.at[i, 'torque_copy']
            match = re.search(pattern, text)

            if match:
                torque_kgm = float(match.group(1))  # Момент силы в kgm
                rpm = int(match.group(2).replace(",", ""))  # Обороты в rpm (без запятых)

                # Перевод момента силы в N·m
                torque_nm = torque_kgm * 9.8

                # Создание результирующей строки
                df_test.at[i, 'torque_copy'] = f"{torque_nm:.2f} Nm @ {rpm} rpm"
        import re
    for i in range(1000):
        if isinstance(df_test.at[i, 'torque_copy'], str):
            pattern = r"([\d.]+)@\s*([\d,]+-[\d,]+)\((\w+)@\s*(\w+)\)"
            text = df_test.at[i, 'torque_copy']
            match = re.search(pattern, text)

            if match:
                torque_kgm = float(match.group(1))  # Момент силы в kgm
                rpm_range = match.group(2).replace(",", "")  # Диапазон оборотов, без запятых

                # Перевод момента силы в N·m
                torque_nm = torque_kgm * 9.8

                # Создание результирующей строки
                df_test.at[i, 'torque_copy'] = f"{torque_nm:.2f} Nm @ {rpm_range} rpm"
    import re
    for i in range(1000):
        if isinstance(df_test.at[i, 'torque_copy'], str):
            pattern = r"([\d.]+)kgm@\s*(\d+)rpm"
            text = df_test.at[i, 'torque_copy']
            match = re.search(pattern, text)

            if match:
                torque_kgm = float(match.group(1))  # Момент силы в kgm
                rpm = int(match.group(2))  # Диапазон оборотов, без запятых

                # Перевод момента силы в N·m
                torque_nm = torque_kgm * 9.8

                # Создание результирующей строки
                df_test.at[i, 'torque_copy'] = f"{torque_nm:.2f} Nm @ {rpm} rpm"
    for i in range(1000):
        if isinstance(df_test.at[i, 'torque_copy'], str):
            pattern = r"([\d.]+)\s*kgm\s*at\s*([\d,]+)\s*rpm"
            text = df_test.at[i, 'torque_copy']

            # Поиск совпадений
            match = re.search(pattern, text)

            if match:
                torque_kgm = float(match.group(1))  # Момент силы в kgm
                rpm = int(match.group(2).replace(",", ""))  # Обороты в rpm (без запятых)

                # Перевод момента силы в N·m
                torque_nm = torque_kgm * 9.8

                # Создание результирующей строки
                df_test.at[i, 'torque_copy'] = f"{torque_nm:.2f} Nm @ {rpm} rpm"
    for i in range(1000):
        if isinstance(df_test.at[i, 'torque_copy'], str):
            # Регулярное выражение
            pattern = r"([\d.]+)\s*Nm\s*at\s*([\d,]+)\s*rpm"
            text = df_test.at[i, 'torque_copy']

            # Поиск совпадений
            match = re.search(pattern, text)

            if match:
                torque_nm = float(match.group(1))  # Момент силы в Nm
                rpm = int(match.group(2).replace(",", ""))  # Обороты в rpm (без запятых)

                # Создание результирующей строки
                df_test.at[i, 'torque_copy'] = f"{torque_nm:.2f} Nm @ {rpm} rpm"
    for i in range(1000):
        if isinstance(df_test.at[i, 'torque_copy'], str):
            pattern = r"([\d.]+)\s*kgm\s*at\s*([\d,-]+)rpm"
            text = df_test.at[i, 'torque_copy']
            match = re.search(pattern, text)
            if match:
                torque_kgm = float(match.group(1))  # Момент силы в kgm
                rpm_range = match.group(2).replace(",", "")  # Диапазон оборотов (без запятых)

                # Перевод момента силы в N·m
                torque_nm = torque_kgm * 9.8

                # Создание результирующей строки
                df_test.at[i, 'torque_copy'] = f"{torque_nm:.2f} Nm @ {rpm_range} rpm"
    for i in range(1000):
        if isinstance(df_test.at[i, 'torque_copy'], str):
            pattern = r"([\d.]+)\s*Nm\s*at\s*([\d,-]+)\s*rpm"
            text = df_test.at[i, 'torque_copy']
            match = re.search(pattern, text)
            if match:
                torque_nm = float(match.group(1))  # Момент силы в Nm
                rpm_range = match.group(2).replace(",", "")  # Диапазон оборотов (без запятых)

                # Создание результирующей строки
                df_test.at[i, 'torque_copy'] = f"{torque_nm:.2f} Nm @ {rpm_range} rpm"
    for i in range(1000):
        if isinstance(df_test.at[i, 'torque_copy'], str):
            pattern = r"([\d.]+)\s*KGM\s*at\s*([\d,-]+)\s*RPM"
            text = df_test.at[i, 'torque_copy']
            match = re.search(pattern, text)
            if match:
                torque_kgm = float(match.group(1))  # Момент силы в KGM
                rpm_range = match.group(2).replace(",", "")  # Диапазон оборотов

                # Перевод момента силы в N·m
                torque_nm = torque_kgm * 9.8

                # Создание результирующей строки
                df_test.at[i, 'torque_copy'] = f"{torque_nm:.2f} Nm @ {rpm_range} rpm"
    df_test.at[880, 'torque_copy'] = '480Nm@2500rpm'
    df_test.at[865, 'torque_copy'] = '400Nm@2000rpm'
    df_test.at[405, 'torque_copy'] = '48Nm@3000rpm'

    df_test[['torque', 'max_torque_rpm']] = df_test['torque_copy'].str.split('@', expand=True)

    df_test.at[440, 'max_torque_rpm'] = 'NaN'
    df_test.at[148, 'max_torque_rpm'] = 'NaN'
    df_test.at[189, 'max_torque_rpm'] = 2375.0

    for i in range(1000):
        if isinstance(df_test.at[i, 'torque'], float):
            pass
        elif ' Nm' in df_test.at[i, 'torque']:
            df_test.at[i, 'torque'] = float(df_test.at[i, 'torque'].replace(' Nm', ''))
        elif 'Nm' in df_test.at[i, 'torque']:
            df_test.at[i, 'torque'] = float(df_test.at[i, 'torque'].replace('Nm', ''))
        elif 'nm' in df_test.at[i, 'torque']:
            df_test.at[i, 'torque'] = float(df_test.at[i, 'torque'].replace('nm', ''))
        elif ' nm' in df_test.at[i, 'torque']:
            df_test.at[i, 'torque'] = float(df_test.at[i, 'torque'].replace(' nm', ''))
        elif 'kgm' in df_test.at[i, 'torque']:
            df_test.at[i, 'torque'] = float(df_test.at[i, 'torque'].replace('kgm', '')) * 9.8
        else:
            print(i)

    df_test.at[793, 'torque'] = 110
    df_test.at[456, 'torque'] = 259.87
    df_test.at[321, 'torque'] = 151
    for i in range(1000):
        if isinstance(df_test.at[i, 'max_torque_rpm'], float):
            pass
        elif 'rpm' in df_test.at[i, 'max_torque_rpm']:
            text = df_test.at[i, 'max_torque_rpm']
            pattern = r"(\d+)-(\d+)rpm"
            match = re.search(pattern, text)
            if match:
                low = int(match.group(1))  # Нижняя граница (1500)
                high = int(match.group(2))  # Верхняя граница (2500)
                df_test.at[i, 'max_torque_rpm'] = (low + high) / 2  # Среднее значение
    for i in range(1000):
        if isinstance(df_test.at[i, 'max_torque_rpm'], float):
            pass
        elif 'rpm' in df_test.at[i, 'max_torque_rpm']:
            text = df_test.at[i, 'max_torque_rpm']
            pattern = r"(\d+)-(\d+)\s*rpm"
            match = re.search(pattern, text)
            if match:
                low = int(match.group(1))  # Нижняя граница (1500)
                high = int(match.group(2))  # Верхняя граница (2500)
                df_test.at[i, 'max_torque_rpm'] = (low + high) / 2  # Среднее значение
    df_test.at[179, 'max_torque_rpm'] = 4500.0
    df_test.at[518, 'max_torque_rpm'] = 4000.0
    df_test.at[625, 'max_torque_rpm'] = 2500.0
    df_test.at[693, 'max_torque_rpm'] = 4000.0
    df_test.at[793, 'max_torque_rpm'] = 4800.0
    df_test.at[885, 'max_torque_rpm'] = 4500.0
    for i in range(1000):
        if isinstance(df_test.at[i, 'max_torque_rpm'], float):
            pass
        elif 'rpm' in df_test.at[i, 'max_torque_rpm']:
            df_test.at[i, 'max_torque_rpm'] = float(df_test.at[i, 'max_torque_rpm'].replace('rpm', ''))
        elif ' rpm' in df_test.at[i, 'max_torque_rpm']:
            df_test.at[i, 'max_torque_rpm'] = float(df_test.at[i, 'max_torque_rpm'].replace(' rpm', ''))
        else:
            print(i)

    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].astype(float)
    median_value = df_test['max_torque_rpm'].median()
    df_test['max_torque_rpm'] = df_test['max_torque_rpm'].fillna(median_value)
    median_value = df_test['seats'].median()
    df_test['seats'] = df_test['seats'].fillna(median_value)
    median_value = df_test['torque'].median()
    df_test['torque'] = df_test['torque'].fillna(median_value)

    df_test['engine'] = df_test['engine'].astype(int)
    df_test['seats'] = df_test['seats'].astype(int)

    df_cleaned = df_test.select_dtypes([int, float])

    X = df_cleaned.drop('selling_price', axis = 1)


    return X

@app.post("/predict/")
async def predict(features: CarFeatures):
    input_data = preprocess_features(features)

    input_data = input_data.reshape(1, -1)

    predicted_price = float(model.predict(input_data)[0])
    # Возврат результата
    return {"predicted_price": predicted_price}

@app.post("/predict_items", response_class=StreamingResponse)
async def predict_items(file: UploadFile = File(...)):
    content = await file.read()
    df1 = pd.read_csv(io.BytesIO(content))
    df2 = df1.copy()
    formatting = DataPreprocessing(df1)
    predict = model.predict(formatting)

    df2['predictions_price'] = predict

    output_stream = io.StringIO()
    df2.to_csv(output_stream, index=False)
    output_stream.seek(0)

    response = StreamingResponse(iter([output_stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions_output.csv"
    return response