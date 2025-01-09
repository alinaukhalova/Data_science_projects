#!/usr/bin/env python
# coding: utf-8

# # Определение стоимости автомобилей

# Сервис по продаже автомобилей с пробегом «Не бит, не крашен» планирует расширить свою клиентскую базу с помощью нового мобильного приложения. Одной из ключевых функций приложения станет возможность расчёта рыночной стоимости автомобиля по его характеристикам. Это поможет пользователям оценить текущую стоимость своей машины на основе реальных рыночных данных, что сделает процесс купли-продажи прозрачнее и удобнее.
# 
# Нам предстоит построить модель, которая будет эффективно предсказывать рыночную стоимость автомобилей на основе их технических характеристик и комплектаций.
# 
# Перед нами стоит несколько задач:
# 
# - Подготовить и проанализировать данные о автомобилях, их характеристиках, комплектациях и ценах.
# - Построить модели машинного обучения, которые смогут предсказывать стоимость автомобиля с высокой точностью.
# - Сравнить модели по трём важным метрикам:
#   1) качество предсказания;
#   2) время обучения модели;
#   3) время предсказания модели.
#   
# Данные, используемые для анализа, включают следующие признаки:
# 
# Категориальные признаки: тип кузова автомобиля (VehicleType), тип коробки передач (Gearbox), модель автомобиля (Model), тип топлива (FuelType), марка автомобиля (Brand) и информация о ремонтах (Repaired)   
# Колличественные признаки: дата загрузки анкеты из базы данных (DateCrawled), год регистрации (RegistrationYear), мощность двигателя (л.с.) (Power), пробег в километрах (Kilometer), месяц регистрации (RegistrationMonth), дата создания анкеты (DateCreated), количество фотографий (NumberOfPictures), почтовый индекс владельца (PostalCode) и дата последней активности пользователя (LastSeen)   
# Целевой признак для построения модели - цена автомобиля (Price).

# ## Подготовка данных

# In[1]:


get_ipython().system('pip install scikit-learn==1.1.3')
get_ipython().system('pip install phik')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
import lightgbm as lgb
import phik
import warnings
from sklearn.compose import (ColumnTransformer, make_column_transformer)
from sklearn.pipeline import (make_pipeline, Pipeline)
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (RandomizedSearchCV, train_test_split, GridSearchCV)
from sklearn.metrics import (mean_squared_error, roc_auc_score)
from catboost import CatBoostRegressor

# настройки
warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# Для начала загрузим наш исходный датасет 

# In[2]:


autos = pd.read_csv('/datasets/autos.csv')


# In[3]:


display(autos.head(5))
autos.info()


# Выведем количество пропусков:

# In[4]:


print(autos.isna().sum())


# Мы сразу можем заметить, что в исходных данных есть пропуски в столбцах VehicleType, Gearbox, Model, FuelType и Repaired, которые требуют обработки. Сначала приведём названия столбцов к нижнему регистру с помощью метода str.lower, затем проверим наличие полных дубликатов и выполним обработку пропущенных значений.

# In[5]:


autos.columns = autos.columns.str.lower()
print(autos.columns)


# In[6]:


print(autos.duplicated().sum())


# Удалим полные дубликаты

# In[7]:


autos = autos.drop_duplicates()
print(autos.duplicated().sum())


# Если удалить все объекты с пропусками, мы потеряем очень много данных. 
# Это может существенно повлиять на качество модели, так как значительная часть информации будет утрачена, что критично для построения точных предсказаний. Выведем уникальные значения, произведем замену пропусков для каждого признака и также проверим категориальные признаки на наличие ошибок

# In[8]:


print(autos['vehicletype'].unique())
print(autos['vehicletype'].value_counts())


# Точно определить категорию для пропущенного значения в 'vehicletype' невозможно, поэтому используем заглушку и заменим пропуски на unknown

# In[9]:


autos.fillna({'vehicletype': 'unknown'}, inplace=True)
print(autos['vehicletype'].value_counts())


# Сделаем то же самое для других признаков. 

# In[10]:


print(autos['gearbox'].unique())
print(autos['gearbox'].value_counts())


# Заменим все неизвестные типы коробки передач на значение unknown

# In[11]:


autos.fillna({'gearbox': 'unknown'}, inplace=True)


# In[12]:


print(autos['model'].unique())


# Модели range_rover и range_rover_sport могут считаться связанными, поскольку обе принадлежат к одной и той же линейке автомобилей Land Rover, поэтому переименуем range_rover_sport просто в range_rover

# In[13]:


autos['model'] = autos['model'].replace(['range_rover_sport'], 'range_rover')


# Как и выше, добавим неизвестные значения моделей, тип топлива и была машина в ремонте или нет в unknown

# In[14]:


autos.fillna({'model': 'unknown'}, inplace=True)


# In[15]:


print(autos['fueltype'].unique())
print(autos['fueltype'].value_counts())


# In[16]:


# переименуем gasoline в petrol, поскольку эти значения оба обозначают "бензин"
autos['fueltype'] = autos['fueltype'].replace(['gasoline'],'petrol')


# In[17]:


autos.fillna({'fueltype': 'unknown'}, inplace=True)


# In[18]:


print(autos['repaired'].unique())
print(autos['repaired'].value_counts())


# In[19]:


autos.fillna({'repaired': 'unknown'}, inplace=True)


# Посмотрим, как обработались все пропуски

# In[20]:


print(autos.isna().sum())


# Посмотрим на все оставшиеся признаки и проанализируем их

# In[21]:


print(autos['brand'].unique())


# Удалим следующие столбцы, которые не будут нужны для построения модели: datecrawled, registrationmonth, datecreated, numberofpictures, postalcode и lastseen. Эти признаки не оказывают влияния на цену автомобиля (наш целевой признак), так как не связаны с его техническими характеристиками или комплектацией. Их сохранение не добавит полезной информации для модели, поэтому они не представляют практической ценности для машинного обучения.

# In[22]:


autos.drop(columns=['datecrawled', 'datecreated', 'numberofpictures','postalcode', 'lastseen', 'registrationmonth'],axis=1,inplace=True)


# In[23]:


display(autos.head(5))


# Для анализа и количественных признаков построим гистрограммы распределений и диаграммы размаха. После этого можно исключить выбросы, чтобы повысить качество данных.

# In[24]:


autos.describe().T


# 1) Цена машины в евро - price

# In[25]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
fig.suptitle('Анализ распределения цены автомобиля', fontsize=18)

# Гистограмма распределения цены
autos['price'].plot(kind='hist', bins=20, ax=axes[0])
axes[0].set_title('Гистограмма распределения цены')
axes[0].set_xlabel('Цена (в евро)')
axes[0].set_ylabel('Количество')

# Диаграмма размаха для цены
autos['price'].plot(kind='box', ax=axes[1])
axes[1].set_title('Диаграмма размаха цены')
axes[1].set_ylabel('Цена (в евро)')

plt.tight_layout()
plt.show()


# Цена автомобиля не может равняться нулю, поэтому необходимо удалить все записи, в которых цена указана от 0 до 50 евро. Это позволяет избежать искажений в анализе и обеспечит более точные результаты при построении модели. Таким образом, мы отфильтруем данные, чтобы оставить только автомобили с действительными ценами, что также поможет улучшить качество предсказания модели в дальнейшем.

# In[26]:


autos = autos.query('price > 50')


# 2) Год регистрации автомобиля - registrationyear

# In[27]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
fig.suptitle('Анализ распределения года регистрации автомобиля', fontsize=18)

# Гистограмма распределения года регистрации
autos['registrationyear'].plot(kind='hist', bins=20, ax=axes[0])
axes[0].set_title('Гистограмма распределения года регистрации')
axes[0].set_xlabel('Год регистрации')
axes[0].set_ylabel('Количество')

# Диаграмма размаха для года регистрации
autos['registrationyear'].plot(kind='box', ax=axes[1])
axes[1].set_title('Диаграмма размаха года регистрации')
axes[1].set_ylabel('Год регистрации')

plt.tight_layout()
plt.show()


# Судя по графикам, в данных есть значения для registrationyear, которые выглядят как выбросы, такие как 1000 и 9999. Эти значения явно выходят за разумные границы (автомобиль не мог быть зарегистрирован до 1800-х годов или после 2016 года). Исключим такие аномалии и оставить только реалистичные значения года регистрации автомобиля, например, от 1960 до 2016 года.

# In[28]:


autos = autos[(autos['registrationyear'] >= 1960) & (autos['registrationyear'] <= 2016)]
print(autos['registrationyear'].describe())


# 3) Мощность автомобиля (л. с.) - power

# In[29]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
fig.suptitle('Анализ распределения мощности автомобиля (л.с.)', fontsize=18)

# Гистограмма распределения мощности
autos['power'].plot(kind='hist', bins=20, ax=axes[0])
axes[0].set_title('Гистограмма распределения мощности')
axes[0].set_xlabel('Мощность (л.с.)')
axes[0].set_ylabel('Количество')

# Диаграмма размаха для мощности
autos['power'].plot(kind='box', ax=axes[1])
axes[1].set_title('Диаграмма размаха мощности')
axes[1].set_ylabel('Мощность (л.с.)')

plt.tight_layout()
plt.show()


# Судя по графикам у мощности автомобилей, имеются явные выбросы, так как значения мощности варьируются от 0 до 20,000 л.с. Обычно, диапазон реальной мощности автомобилей составляет от 40 до 650 л.с. для легковых автомобилей. Такие значения, как 0 л.с. и 20,000 л.с., могут свидетельствовать об ошибках ввода данных или неточностях. Укажем значения мощности меньше 650 л.с. но больше 40 л.с.

# In[30]:


autos = autos[(autos['power'] > 40) & (autos['power'] <= 650)]


# 4) Пробег автомобиля (км) - kilometer

# In[31]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
fig.suptitle('Анализ распределения пробега автомобиля (км)', fontsize=18)

# Гистограмма распределения пробега
autos['kilometer'].plot(kind='hist', bins=20, ax=axes[0])
axes[0].set_title('Гистограмма распределения пробега')
axes[0].set_xlabel('Пробег (км)')
axes[0].set_ylabel('Количество')

# Диаграмма размаха для пробега
autos['kilometer'].plot(kind='box', ax=axes[1])
axes[1].set_title('Диаграмма размаха пробега')
axes[1].set_ylabel('Пробег (км)')

plt.tight_layout()
plt.show()


# In[32]:


autos['kilometer'].value_counts()


# Согласно данным и этим графикам, можно заметить, что пробег большинства автомобилей сконцентрирован на уровне 150,000 км, что может указывать на то, что это распространённый пробег для подержанных автомобилей в нашем наборе данных. Оставим все данные по километражу и позволим модели лучше учесть особенности рынка подержанных автомобилей, тем самым улучшим точность прогнозов.

# После анализа признаков можно провести корреляционный анализ для оценки взаимосвязей между ними. Построим матрицу корреляции коэффициента Фи

# In[33]:


# построим матрицу корреляции коэффициента Фи'

selected_columns = ['price', 'vehicletype', 'registrationyear', 'gearbox', 'power', 'model', 'kilometer', 'fueltype', 'brand', 'repaired']
autos_subset = autos[selected_columns]

# Вычисляем матрицу корреляции с помощью phik
autos_phik = autos_subset.phik_matrix()

# Визуализация матрицы корреляции
plt.figure(figsize=(12, 10))
sns.heatmap(autos_phik, annot=True, fmt='.2f', cmap='PuBu', cbar=True)
plt.title('Матрица корреляции коэффициента $\phi_K$', fontsize=15)
plt.show()


# Мы сразу можем заметить, что между признаками model и brand наблюдается полная корреляция (1.00). Это указывает на то, что наши данные пересекаются: для каждой модели соответствует только один конкретный бренд. Также признак model с признаком vehicletype имеет высокую корреляцию (0.898), что, вероятно, связано с тем, что большинство моделей автомобилей выпускаются в определенном варианте кузова, создавая взаимосвязь между этими признаками. Поэтому лучше удалить признак model, так как brand уже достаточно полно передаёт информацию о моделях, что поможет избежать избыточности данных и уменьшить риск мультиколлинеарности.
# 
# Кроме того, между ценой price и мощностью power автомобиля существует умеренная положительная корреляция (0.5), что указывает на то, что увеличение мощности автомобиля связано с повышением его цены. Значит, автомобили с более мощными двигателями часто имеют более высокую стоимость.
# 
# Также между ценой price и годом регистрации registrationyear заметная корреляция и она составляет 0.66, что скорее всего говорит нам о том, что более новые автомобили, как правило, стоят дороже.
# 
# Как и у model, мы видим заметную корреляцию между признаками brand и vehicletype (0.59) и она может быть объяснена тем, что определенные бренды выпускают машины, которые чаще представлены в определенных типах кузова. Вдобавок, умеренная корреляция между brand и power (0.55) и между brand и gearbox (0.52) говорит о том, что определенные марки имеют тенденцию к производству автомобилей с определенной мощностью и коробкой передач.
# 
# Оставшиеся признаки демонстрируют либо немного заметные, или же слабо выраженные корреляционные связи между друг другом.

# In[34]:


# уберем признак model
autos.drop(columns=['model'],axis=1,inplace=True)


# После всех преобразований опять перепроверим на дубликаты, и если они есть, удалим их

# In[35]:


print(autos.duplicated().sum())


# In[36]:


autos = autos.drop_duplicates()
print(autos.duplicated().sum())


# Итак, по итогу 1 шага работы можно сделать небольшие выводы:
# Мной были выполнены несколько этапов, которые обеспечили качество и полноту набора данных для дальнейшего анализа и построения модели. 
# 
# Для начала мы загрузили и провели первичный анализ данных. Датасет был загружен, и мы выявили пропуски в нескольких столбцах: VehicleType, Gearbox, Model, FuelType и Repaired. Для обработки пропусков мы заменили их на значение "unknown", что позволило сохранить структуру данных без значительной потери информации.
# После мы обнаружили и удалили 4 полных дубликата, чтобы избежать искажения результатов анализа.
# Далее были исключены столбцы, которые не влияют на наш целевой признак на цену автомобиля (datecrawled, registrationmonth, datecreated, numberofpictures, postalcode и lastseen). Эти признаки не имели практической ценности для модели, и их сохранение могло бы лишь усложнить процесс машинного обучения.
# 
# Следующим шагом мы провели анализ количественных признаков и выявили аномалии в некоторых данных:
#  - Цена автомобиля (price): Удалены записи с ценой меньше 50 евро, чтобы избежать искажений в анализе.
#  - Год регистрации (registrationyear): Исключены нереалистичные значения года регистрации машины (например, 1000 и 9999), что позволило оставить только приемлемые данные (от 1960 года до 2016).
#  - Мощность автомобиля (power): Мощность автомобиля мы указали в диапазоне от 40 л.с. до 650 л.с.
# 
# В заключение, мы построили матрицу корреляции Фи и провели корреляционный анализ, который выявил следующие результаты: мы обнаружили полную корреляцию (1.00) между признаками model и brand, что указывает на дублирование данных: каждой модели соответствует только один бренд. Также присутствовала высокая корреляция (0.898) между model и vehicletype была связана с тем, что большинство моделей выпускались в определённых вариантах кузова. Поэтому мы решили удалить признак model, поскольку brand уже достаточно отражал информацию о моделях и помогал избежать избыточности. 
# Также мы обнаружили умеренную положительную корреляцию (0.5) между ценой price и мощностью power, что указывает на то, что более мощные автомобили, как правило, имеют более высокую стоимость. Заметная корреляция (0.66) между price и годом регистрации registrationyear свидетельствовала о том, что более новые автомобили, как правило, стоили дороже. Кроме того, присутствовала корреляция (0.59) между brand и vehicletype, что объясняется тем, что определённые бренды производят автомобили, соответствующие определённым типам кузова. Умеренные корреляции (0.55) между brand и power, а также (0.52) между brand и gearbox указывали на то, что некоторые марки выпускают автомобили с определённой мощностью и типом коробки передач. Остальные признаки показывают либо незначительные, либо слабые корреляционные связи между собой.
# И в конце мы перепроверили наши данные на наличие дубликатов с последующим их удалением.
# 
# В результате проведенных действий мы подготовили данные для дальнейшего анализа и построения предсказательной модели. Проведенные шаги обеспечили целостность и качество набора данных, что создает надежную основу для эффективного анализа и более точных предсказаний в будущем.

# ## Обучение моделей

# Далее для начала необходимо разделить датасет на тренировочную и тестовую выборки.

# In[37]:


# Разделим данные на признаки и целевой признак
X = autos.drop('price', axis=1)
y = autos['price']


# In[38]:


# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# **1. Модель LightGBM**

# In[39]:


X_train_lgbm = X_train.copy()
y_train_lgbm = y_train.copy()
X_test_lgbm = X_test.copy()
y_test_lgbm = y_test.copy()


# In[40]:


cat_features = ['vehicletype', 'gearbox', 'fueltype', 'brand', 'repaired']
for col in cat_features:
    X_train_lgbm[col] = X_train_lgbm[col].astype('category')
    X_test_lgbm[col] = X_test_lgbm[col].astype('category')


# In[41]:


model_1 = lgb.LGBMRegressor(random_state=RANDOM_STATE)

param_grid = {'num_leaves': [100,150],'learning_rate':[0.2, 0.4, 0.5, 0.8]}

grid_search_lgbm=GridSearchCV(estimator=model_1,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=3,
                          verbose=3)

grid_search_lgbm.fit(X_train_lgbm, y_train_lgbm)

lgbm_params = grid_search_lgbm.best_params_

# Вывод лучших параметров и значения метрики
print()
print("Лучшие параметры:", lgbm_params)
print("Метрика RMSE, полученная с помощью кросс валидации:", (grid_search_lgbm.best_score_ * -1))


# In[42]:


get_ipython().run_cell_magic('time', '', '# Засекаем время обучения модели на всей тренировочной выборке с лучшими параметрами\nstart = time.time()\n\nlgbm_model = lgb.LGBMRegressor(num_leaves = lgbm_params[\'num_leaves\'], learning_rate = lgbm_params[\'learning_rate\'], random_state=RANDOM_STATE)\nlgbm_model.fit(X_train_lgbm, y_train_lgbm)\n\n# Время обучения\nend = time.time()\ntime_lgbm_fit =end-start\nprint(f"Время обучения модели LightGBM на тренировочной выборке: {time_lgbm_fit:.2f} секунд")\n')


# In[43]:


get_ipython().run_cell_magic('time', '', '# Время предсказания модели LightGBM\nstart_time = time.time()\ny_pred_lgbm = lgbm_model.predict(X_test_lgbm)\nend_time = time.time()\nprediction_time_lgbm = end_time - start_time\nprint(f"Время предсказания модели LightGBM: {prediction_time_lgbm:.2f} секунд")\n')


# Лучшие параметры LightGBM: learning_rate: 0.2, num_leaves: 150. Качество модели на кросс-валидации: 1653.53. Найденое время обучение и предсказания, а также и этот качество модели на кросс-валидации будем в дальнейшем сравнивать с другими моделями, чтобы выбрать лучшую модель по трем критериям.

# **2. Модель CatBoostRegressor**

# In[44]:


X_train_cbr = X_train.copy()
y_train_cbr = y_train.copy()
X_test_cbr = X_test.copy()
y_test_cbr = y_test.copy()


# In[45]:


cat_features = ['vehicletype', 'gearbox', 'fueltype', 'brand', 'repaired']
for col in cat_features:
    X_train_cbr[col] = X_train_cbr[col].astype('category')
    X_test_cbr[col] = X_test_cbr[col].astype('category')


# In[46]:


model_2 = CatBoostRegressor(iterations=100, verbose=100)

param_grid_cbr = {
    'learning_rate': [0.1, 0.4, 0.9],
    'random_state': [42],
    'depth': [6, 8]
}

random_search_cbr = RandomizedSearchCV(
    model_2, 
    param_distributions=param_grid_cbr, 
    scoring='neg_root_mean_squared_error', 
    random_state=RANDOM_STATE,
    n_iter=3,
    cv=5,
    n_jobs=-1
)

random_search_cbr.fit(X_train_cbr, y_train_cbr, cat_features=cat_features)

cbr_params = random_search_cbr.best_params_

# Вывод лучших параметров и значения метрики
print()
print("Лучшие параметры:", cbr_params)
print("Метрика RMSE, полученная с помощью кросс валидации:", (random_search_cbr.best_score_ * -1))


# In[47]:


get_ipython().run_cell_magic('time', '', '# Засекаем время обучения модели на всей тренировочной выборке с лучшими параметрами\nstart = time.time()\n\ncbr_model = CatBoostRegressor(random_state = cbr_params[\'random_state\'], learning_rate = cbr_params[\'learning_rate\'], depth = cbr_params[\'depth\'], iterations=100, verbose=100)\ncbr_model.fit(X_train_cbr, y_train_cbr, cat_features=cat_features)\n\n# Время обучения\nend = time.time()\ntime_cbr_fit =end-start\nprint(f"Время обучения модели CatBoostRegressor на тренировочной выборке: {time_cbr_fit:.2f} секунд")\n')


# In[48]:


get_ipython().run_cell_magic('time', '', '# Время предсказания модели CatBoostRegressor\nstart_time = time.time()\ny_pred_cbr = cbr_model.predict(X_test_cbr)\nend_time = time.time()\nprediction_time_cbr = end_time - start_time\nprint(f"Время предсказания модели CatBoostRegressor: {prediction_time_cbr:.2f} секунд")\n')


# Мы провели обучение модели CatBoostRegressor с использованием оптимальных гиперпараметров: random_state установлен в 42, learning_rate равен 0.9, а depth составляет 8. Качество модели на кросс-валидации: 1725.89. Найденое время обучение и предсказания, а также и этот качество модели на кросс-валидации будем в дальнейшем сравнивать с другими моделями, чтобы выбрать лучшую модель по трем критериям.

# **3. Модель LinearRegression**

# In[49]:


X_train_lr = X_train.copy()
y_train_lr = y_train.copy()
X_test_lr = X_test.copy()
y_test_lr = y_test.copy()


# In[50]:


ohe_columns = ['vehicletype', 'gearbox', 'fueltype', 'brand', 'repaired']  
num_columns = ['registrationyear', 'power', 'kilometer']


# In[51]:


ohe_columns = ['vehicletype', 'gearbox', 'fueltype', 'brand', 'repaired']
for col in ohe_columns:
    X_train_lr[col] = X_train_lgbm[col].astype('object')
    X_test_lr[col] = X_test_lgbm[col].astype('object')


# In[52]:


# создадим пайплайн для OneHotEncoder:
ohe_pipe = Pipeline(
    [
        (
            'simpleImputer_ohe', 
            SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ),
        (
            'ohe', 
            OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False))
    ]
)


# In[53]:


# Преобразователи для числовых данных
num_pipe = Pipeline([
    ('simpleImputer_num', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])


# In[54]:


data_preprocessor = ColumnTransformer(
    [
        ('ohe', ohe_pipe, ohe_columns),
        ('num', num_pipe, num_columns)
    ], 
    remainder='passthrough'
)


# In[55]:


pipe_final_lr = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('model', LinearRegression())
    ]
)


# In[56]:


# Гиперпараметры для LinearRegression
param_grid_lr = {
    'model': [LinearRegression()],
    'model__fit_intercept': [True, False],
    'model__copy_X': [True, False],
    'model__n_jobs': [1, -1]
}

random_search_lr = RandomizedSearchCV(
    pipe_final_lr, 
    param_distributions=param_grid_lr, 
    cv=5,
    scoring='neg_root_mean_squared_error', 
    random_state=RANDOM_STATE,
    n_iter=8,
    n_jobs=-1
)

random_search_lr.fit(X_train_lr, y_train_lr)

lr_params = random_search_lr.best_params_

# Вывод лучших параметров и значения метрики
print()
print("Лучшие параметры:", lr_params)
print("Метрика RMSE, полученная с помощью кросс валидации:", (random_search_lr.best_score_ * -1))


# In[57]:


get_ipython().run_cell_magic('time', '', '# Засекаем время обучения модели на всей тренировочной выборке с лучшими параметрами\nstart = time.time()\n\n# Создаем переменную с моделью с лучшими параметрами\nlr_model = random_search_lr.best_estimator_\n\nlr_model.fit(X_train_lr, y_train_lr)\n\n# Время обучения\nend = time.time()\ntime_lr_fit =end-start\nprint(f"Время обучения модели LinearRegression на тренировочной выборке: {time_lr_fit:.2f} секунд")\n')


# In[58]:


get_ipython().run_cell_magic('time', '', '# Время предсказания модели LinearRegression\nstart_time = time.time()\ny_pred_lr = lr_model.predict(X_test_lr)\nend_time = time.time()\nprediction_time_lr = end_time - start_time\nprint(f"Время предсказания модели LinearRegression: {prediction_time_lr:.2f} секунд")\n')


# Мы провели обучение модели LinearRegression с использованием оптимальных гиперпараметров: fit_intercept установлен в True, copy_X равен True, а n_jobs составляет 1. Качество модели на кросс-валидации: 2831.76. Найденное время обучения и предсказания, а также это качество модели на кросс-валидации будут в дальнейшем сравниваться с другими моделями, чтобы выбрать лучшую модель по трем критериям.

# **4. Модель DecisionTreeRegressor**

# In[59]:


X_train_dt = X_train.copy()
y_train_dt = y_train.copy()
X_test_dt = X_test.copy()
y_test_dt = y_test.copy()


# In[60]:


pipe_final_dt = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('model', DecisionTreeRegressor(random_state=RANDOM_STATE))
    ]
)


# In[61]:


# Гиперпараметры для DecisionTreeRegressor
param_grid_dt = {
    'model__max_depth': [5, 10, 20], 
    'model__min_samples_split': [2, 5]
}

random_search_dt = RandomizedSearchCV(
    pipe_final_dt, 
    param_distributions=param_grid_dt, 
    scoring='neg_root_mean_squared_error', 
    random_state=RANDOM_STATE,
    n_iter=5,  
    cv=3,     
    n_jobs=-1  
)

random_search_dt.fit(X_train_dt, y_train_dt)

dt_params = random_search_dt.best_params_

# Вывод лучших параметров и значения метрики
print()
print("Лучшие параметры:", dt_params)
print("Метрика RMSE, полученная с помощью кросс валидации:", (random_search_dt.best_score_ * -1))


# In[62]:


get_ipython().run_cell_magic('time', '', '# Засекаем время обучения модели на всей тренировочной выборке с лучшими параметрами\nstart = time.time()\n\n# Создаем переменную с моделью с лучшими параметрами\ndt_model = random_search_dt.best_estimator_\n\ndt_model.fit(X_train_dt, y_train_dt)\n\n# Время обучения\nend = time.time()\ntime_dt_fit =end-start\nprint(f"Время обучения модели DecisionTreeRegressor на тренировочной выборке: {time_dt_fit:.2f} секунд")\n')


# In[63]:


get_ipython().run_cell_magic('time', '', '# Время предсказания модели DecisionTreeRegressor\nstart_time = time.time()\ny_pred_dt = dt_model.predict(X_test_dt)\nend_time = time.time()\nprediction_time_dt = end_time - start_time\nprint(f"Время предсказания модели DecisionTreeRegressor: {prediction_time_dt:.2f} секунд")\n')


# Мы обучили модель DecisionTreeRegressor, используя оптимальные гиперпараметры: min_samples_split установлен на 5, а max_depth составляет 20. Качество модели на кросс-валидации составило 1979.04. Найденное время обучения и предсказания, а также это значение метрики будут сравниваться с другими моделями, чтобы выбрать наилучший вариант по трем критериям.

# Итак, в этой части работы мы обучили несколько моделей машинного. Для каждой из моделей были проведены подбор и настройка гиперпараметров, что позволило улучшить качество предсказаний. В итоге нами были рассмотрены следующие модели: LightGBM, LinearRegression, DecisionTreeRegressor и CatBoostRegressor. Мы использовали кросс-валидацию для нахождения оптимальных гиперпараметров модели и определения ее качества. Затем мы измерили время обучения модели на всей тренировочной выборке. После этого также замерили время предсказания модели на тестовой выборке, однако результаты тестирования не выводили на экран. Теперь мы можем приступать к анализу всех 4х моделей и выбору лучшей из них.

# ## Анализ моделей

# Проанализируем время обучения, время предсказания и качество моделей.

# In[64]:


lr_rmse_train = random_search_lr.best_score_ * -1
lgbm_rmse_train = grid_search_lgbm.best_score_ * -1
cbr_rmse_train = random_search_cbr.best_score_ * -1
dt_rmse_train = random_search_dt.best_score_ * -1


# In[65]:


# Создание DataFrame с нашими результатами
results = pd.DataFrame({
    'Модель': ['Linear Regression', 'LightGBM', 'CatBoostRegressor', 'DecisionTreeRegressor'],
    'RMSE': [lr_rmse_train, lgbm_rmse_train, cbr_rmse_train, dt_rmse_train],
    'Время предсказания (сек)': [prediction_time_lr, prediction_time_lgbm, prediction_time_cbr, prediction_time_dt],
    'Время обучения (сек)': [time_lr_fit, time_lgbm_fit, time_cbr_fit, time_dt_fit]  
})

# Вывод результатов
results


# В данной таблице представлены результаты оценки качества и времени работы 4х моделей для прогнозирования. Мы сравнили модели по метрике RMSE и времени, затрачиваемому на обучение и предсказание. Далее определим, какая из них наиболее эффективена для нашей задачи. 
#  - **Linear Regression**: RMSE равно 2831.76 — это наибольший результат среди всех моделей, что указывает на низкое качество предсказаний. Время предсказания достоточно быстрое, что делает модель подходящей для задач, где требуется быстрая обработка. Также у модели относительно быстрое время обучения.
#  - **LightGBM**: RMSE равно 1653.54 — это лучшая метрика среди всех моделей, показывающая высокую точность предсказаний. Время предсказания и время обучения больше, чем у Linear Regression, но все еще показатели остаются приемлемыми.
#  - **CatBoostRegressor**: RMSE равно 1725.89 — хорошее качество предсказаний, хотя и немного хуже, чем у LightGBM. Наименьшее время предсказания, что делает его очень эффективным. Время обучения сопоставимо с LightGBM, что может быть приемлемым для сложных задач.
#  - **DecisionTreeRegressor**: RMSE равно 1979.04 — качество предсказаний находится между Linear Regression и CatBoost. Время предсказания среднее, больше, чем у CatBoost, но меньше, чем у LightGBM. Самое быстрое время обучения среди всех моделей.
#  
# LightGBM показывает наилучшие показатели по метрике RMSE, что указывает на его способность эффективно моделировать данные. Хотя время обучения и предсказания выше, чем у некоторых других моделей, его высокая точность делает его предпочтительным выбором в случаях, когда критически важно качество предсказаний. Это особенно важно в сценариях, где ошибки могут иметь серьезные последствия, поэтому предпочтение отдается моделям, которые обеспечивают максимальную надежность.
# 
# Таким образом, LightGBM следует выбрать в качестве финальной модели, так как она продемонстрировала отличные результаты по всем трем параметрам, обеспечивая оптимальное сочетание качества предсказаний и эффективности. 
# 
# Теперь посмотрим значения RMSE у нашей лучшей модели на тестовой выборке

# In[66]:


# Оценка качества модели LightGBM на тестовой выборке
rmse_test_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
print("RMSE LightGBM на тестовой выборке: %.2f" % rmse_test_lgbm)


# Пониженное значение RMSE на тестовой выборке по сравнению с кросс-валидацией может указывать на то, что модель не только хорошо подходит для данных, на которых она обучалась, но и способна точно предсказывать на новых данных. Это является положительным признаком и свидетельствует о хорошем качестве модели.
# Небольшая разница между значениями RMSE (1653.53 на тренировочных данных и 1635.29 на тестовой выборке) также говорит о том, что модель не переобучена и сохраняет свою эффективность на новых данных.

# В данном шаге нашей работы мы провели оценку времени обучения, времени предсказания и качества четырех моделей, используемых для прогнозирования. Мы проанализировали такие модели, как LightGBM, LinearRegression, DecisionTreeRegressor и CatBoostRegressor. Результаты представлены в таблице, в которой сравниваются метрики RMSE и затраты времени на обучение и предсказание. На основе полученных данных мы определили наиболее эффективную модель для нашей задачи. 
#   
# Linear Regression: качество модели на кросс-валидации — 2831.76, что указывает на низкое качество предсказаний. Время предсказания быстрое, что может быть полезно, время обучения также относительно короткое.
#   
# LightGBM: качество модели на кросс-валидации — 1653.54, лучший результат среди моделей, демонстрирующий высокую точность предсказаний. Время предсказания и обучения выше, чем у линейной регрессии, но значения все еще приемлемы.
#   
# CatBoostRegressor: качество модели на кросс-валидации — 1725.89, хорошее качество предсказаний, хотя ниже, чем у LightGBM. Обладает наименьшим временем предсказания, а время обучения сопоставимо с LightGBM.
# 
# DecisionTreeRegressor: качество модели на кросс-валидации — 1979.04, качество предсказаний между Linear Regression и CatBoost. Время предсказания среднее: больше, чем у CatBoost и Linear Regression, но меньше, чем у LightGBM. Демонстрирует самое быстрое время обучения среди всех моделей.
# 
# **LightGBM** была выбрана в качестве окончательной лучшей модели, так как она показала превосходные результаты по всем трем критериям, обеспечивая оптимальное сочетание качества предсказаний и эффективности.
# 
# И в конце мы проанализировали значения RMSE для нашей лучшей модели на тестовой выборке. Пониженное значение RMSE на тестовой выборке по сравнению с кросс-валидацией свидетельствовало о том, что модель хорошо адаптировалась к данным, на которых она обучалась, и могла точно предсказывать на новых данных. Это является положительным показателем, подтверждающим высокое качество модели. Небольшая разница между значениями RMSE (1653.53 на тренировочных данных и 1635.29 на тестовой выборке) также подтверждала, что модель не переобучена и сохраняет свою эффективность на новых данных.

# ## Итоговый вывод

# Итак, мы успешно выполнили несколько ключевых задач в рамках нашего проекта, что позволило нам создать и найти  эффективную модель для предсказания рыночной стоимости автомобилей. 
# 
# Для начала мы подготовили и проанализировали данные о автомобилях, их характеристиках, комплектациях и ценах. Мы загрузили данные, провели первичный анализ данных о характеристиках автомобилей, обработали пропуски. Пропуски были заменены на значение "unknown", что помогло сохранить структуру данных без потери информации. Затем мы исключили столбцы, не влияющие на целевой признак, чтобы обеспечить качество и целостность набора данных. В процессе подготовки данных нами также был проведен анализ различных признаков, в ходе которого выявили аномалии, после чего мы исключили или заменили нереалистичные значения и  оставили только те диапазоны значений, которые выглядят правдоподобно. После анализа признаков мы провели корреляционный анализ для оценки взаимосвязей между ними. Построили матрицу корреляции коэффициента Фи для того, чтобы определить признаки, которые сильно коррелируют друг с другом. Убрав сильно коррелирующий признак model, мы улучшили качество модели избежав мультиколлинеарности. И в конце, после всех преобразований, мы удалили полные дубликаты, чтобы избежать искажения результатов анализа. 
# Все эти шаги помогли улучшить качество данных и создать надежную базу для дальнейшего анализа. 
# 
# Для каждой модели были выполнены подбор и настройка гиперпараметров, что способствовало улучшению качества предсказаний. В результате мы рассмотрели следующие модели: LightGBM, LinearRegression, DecisionTreeRegressor и CatBoostRegressor. Мы применили кросс-валидацию для определения оптимальных гиперпараметров и оценки качества моделей. Затем замерили время обучения каждой модели на всей тренировочной выборке, а также время предсказания 
# 
# И потом мы сравнили модели по трем важным метрикам: качество модели на кросс-валидации (RMSE), время обучения модели, время предсказания модели.
# 
# Мы проанализировали все модели и выявили, что модель Linear Regression имеет самое большое значение RMSE, поэтому она не подходит нам для решения поставленной задачи. CatBoostRegressor достигла хорошего качества RMSE, что свидетельствует о хорошем уровне предсказаний, хотя она несколько уступает LightGBM. Эта модель обладает наименьшим временем предсказания, а время обучения сопоставимо с LightGBM. DecisionTreeRegressor показала качество на уровне 1979.04, что находится между Linear Regression и CatBoost. Время предсказания средней продолжительности: больше, чем у CatBoost и линейной регрессии, но меньше, чем у LightGBM. Эта модель демонстрирует самое быстрое время обучения.
# 
# По итогу, модель LightGBM продемонстрировала лучшие результаты, достигнув минимального значения качества модели на кросс-валидации = 1653.54, что сделало её идеальным вариантом для решения этой задачи. Время обучения оказалось умеренным, что обеспечило отличное сочетание точности и эффективности. 
# 
# И в заключение, мы рассмотрели значения RMSE нашей лучшей модели на тестовой выборке. Более низкий RMSE на тестовой выборке по сравнению с кросс-валидацией свидетельствовал о том, что модель хорошо адаптировалась к обучающим данным и способна делать точные предсказания на новых данных. Это является положительным знаком, подтверждающим высокое качество модели. Кроме того, небольшая разница между RMSE (1653.53 на тренировочных данных и 1635.29 на тестовой выборке) подтверждала, что модель не переобучена и сохраняет свою эффективность на новых данных. Таким образом, модель LightGBM способна наиболее точно прогнозировать рыночную стоимость автомобилей, основываясь на их технических характеристиках и комплектациях.
