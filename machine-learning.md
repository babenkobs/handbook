# Machine Learning

## Numpy

import numpy as np

```
mylist = [1,2,3]

myarr = np.array(mylist)
```
Создать массив от 0 до 100 с шагом 20
```
np.arange(0, 101, 20)
```
Получить массив из равноудаленных чисел от 0 до 5 (21 число)
```
np.linspace(0, 5, 21)
```
Получить список из 5-ти случайных целых чисел в двух строках (каждой) от 1 до 70
```
np.random.randint(1, 70, (2, 5))
```
Получить список случайных зафиксированных чисел
```
np.random.seed(42)
np.random.rand(4)
```
Создать одномерный массив и изменить его в двумерный с 5 строками и с 5 столбцами
```
arr = np.arange(0, 25)
arr.reshape(5,5)
```
Найти максимальное и минимальное число
```
arr.max()
arr.min()
```
Получить индекс максимального и минимального значения
```
arr.rangmax()
arr.rangmin()
```
Посмотреть форму массива (кол-во столбцов и строк)
```
arr.shape
```
Получить числа  из массива сначала до индекса 5 (последний не включается)
```
arr[:5]
```
Присвоить элементам массива от 5 до последнего число 66
```
arr[5:] = 66
```
Чтобы присвоить массив другой переменно, нужно использовать метод copy
Если просто присвоить то это присвоит ссылки и изменения в одном массиве, применяться к другому.
```
arr_copy = arr.copy()
```

Получить массив чисел где число больше 4 из другого массива
```
arr2 = arr1[arr1>4]
```

Сумма строк и сумма колонок
```
arr.sum(axis=0) #сумма колонок
arr.sum(axis=1￼) #сумма строк￼
```
Создать массив из 10 нулей
```
np.zeros(10)
```

Сделать абсолютные значения в массиве
```
np.abs(arr)
```

## Pandas
```
import pandas as pd
```

Информация о таблице
```
df.info()
df.C().transpose()
```
Перевернуть таблицу (поменять местами строки со столбцами)
```
df = df.T # или df.transpose()
```
Создать именованный список
```
personal_data = pd.Series(data=[26,33], index=[‘Ivan’, ‘Maria’])
```
Узнать ключи массива
```
personal_data.keys()
```
Сложить два массива и отсутсвующие значения заменить 0
```
sales_1.add(sales_2, fill_value=0)
```
Создание dataframe
```
df = pd.DataFrame(data=mydata, index=myindex, columns=columns_names)
df = pd.DataFrame({“age”:[25,35,45], “isMale”:[True, False, True]})
```
Создание dataframe из csv файла
```
df=pd.read_csv('/home/bobmarley/UNZIP_ME_FOR_NOTEBOOKS_ML_RUS_V1/UNZIP_ME_FOR_NOTEBOOKS_ML_RUS_V1/03-Pandas/tips.csv')
```
Узнать названия колонок и строк
```
df.columns
df.index
```
Показать первые или последние десять строк dataframe
```
df.head(10)
df.tail(10)
```
Получить общую статистику по dataframe
```
df.describe()
```
Получить среднее значение в столбце
```
df[‘price’].mean()
```

Вывести два столбца из df
```
df[[‘total_bill’,’tip’]]
```

Удалить колонку price
```
df = df.drop(‘price’, axis=1)
```

Установить колонку Payment_ID в качестве индекса строк (названия строк)
```
df.set_index(‘Payment_ID’)
```

Отменить индекс
```
df = df.reset_index()
```

Доступ к строке по номеру и индексу
```
df.iloc[0] или df.iloc[0:5]
df.loc[‘Marc’] или df.loc[[‘Marc’, ‘Jack’]]
```

Добавить строку в dataframe
```
df = df.append(one_row)
```

Отфильтровать строки по условию, где в столбце Pop значение больше 50
```
df[df[“Pop”] > 50]
```

В условиях AND это &, а OR это |. Условия записываются в скобки
```
df[(df[“Pop”] > 50) & (df[“Sex”] == “Male”])]
```

Полезный метод проверки есть ли одно из значений
```
df[df[‘Day’].isin([‘Saturday’, ‘Sunday’])]
```

Применить функцию к значениям столбца
```
def someFunc(val):
	return f’Value = {val}’

df[“new_column”] = df[“some_column”].apply(someFunc)

```
Применить функцию к нескольким столбцам
```
def quality(total_bill,tip):
    if tip/total_bill  > 0.25:
        return "Generous"
    else:
        return "Other"

df['Tip Quality'] = df[['total_bill','tip']].apply(lambda df: quality(df['total_bill'],df['tip']),axis=1)

```
Или сделать это быстрее и проще
```
df['Tip Quality'] = np.vectorize(quality)(df['total_bill'], df['tip'])
```

Передать несколько значений в функцию (помимо строки)
```
df['50%'] = df.apply(MyFunc, args=['50%', 100], axis=1)
```

Сортировка по значениям (колонки)
```
df.sort_values(‘tip’)
df.sort_values(‘tip’, ascending=False) #обратная сортировка
df.sort_values([‘tip’,’size’], ascending=False)  #сортировка по двум колонкам (сначала tip, затем по size)
```

Найти максимальное и минимальное значение
```
df[‘total_bill’].max()
df[‘total_bill’].idxmax() - получить index строки с максимальным значением
df[‘total_bill’].min()
df[‘total_bill’].idxmin() - получить index строки с миниимальным значением
```
Найти корреляцию
```
df.corr()
df.corr().quality #отобразить корреляцию только относительно стобца quality
```

Посчитать кол-во строк со значением (категоризация)
```
df[‘some_column’].value_counts()
df[‘some_column’].unique() # массив уникальных значений
df[‘some_column’].nunique() # кол-во уникальных значений
```

Замена значений в колонке
```
df[‘some_column’].replace(‘Value’, ‘V’)
df[‘some_column’].replace([‘Value’, ‘Value2’], [‘V’, ‘V2’])

replace = {‘Value’:’V’, ‘Value2’:’V2’}
df[‘some_column’].map(replace)
```

Отобразить дубликаты
```
df.duplicated()
df.drop_duplicates() #удалить дубликаты
```

Взять диапазон
```
df[‘some_column’].between(10,20, inclusive=True) # с включение верхней границы значений (20)
```

Отсортировать и показать первые 10 строк
```
df.nlargest(10,’tip’)
df.nsmallest(10,’tip’) # сортировка по наименьшему
```

Получить 5 случайных строк из таблицы
```
df.sample(5)
df.sample(frac=0.1) # получить 10% строк из таблицы
```

Сделать dummies переменные из категориальных значений
```
pd.get_dummies(df_new, drop_first=True) #drop_first - оставляет по минимуму колонок
```

Цикл по датафрейму по строкам
```
for idx, row in df.iterrows():
        makeMoney(row)
```



## Отсутствующие данные в Pandas
Показать отсутсвующие значения
```
df.isnull()
df.notnull()
df[df['first_name'].notnull()]
```

Удалить
```
df = df.dropna() #Удалить все строки где есть хотя бы одно неопределенное значение
df.dropna(tresh=1) #удалить строки с неопределенными значениями, но оставить те, где есть хотябы одно определенное значение
df.dropna(subset=[‘last_name’]) #удалить те строки, где неопределенные значения в стобце last_name
df.dropna(axis=1) #удалить столбцы, где есть хоть одно значение
df.drop('a', inplace=True, axis=1) # удалить столбец а
```

Заменить
```
df.fillna(‘new value’) #заменить везде (очень редкий случай)
df[‘price’] = df[‘price’].fillna(0) #заменить на ноль в столбце price
```

Группировка
```
df.groupby(‘year’).mean() #группировать по году, а остальные колонки усреднить
df.groupby([‘year’, ‘speed’]).mean() #группировать две колонки (мультииндекс)
```

Достать строки из мультииндекса
```
df.xs(key=120, level=’speed’) #получить строки со значением 120 в столбце speed (обычно это применяется для второго индекса, так для первого достаточно loc
```

Агрегация
```
df.agg({‘speed’: [‘max’, ‘mean’], weight: [‘mean’, ‘std’]})
```

Обьединение датафреймов
```
pd.concat([df1, df2], axis=1) #обьеденить по столбцам (добавляются столбцы)
pd.concat([df1, df2], axis=0) #обьеденить по строкам (добавляются строки)
```

Merge
```
pd.merge(table1, table2, how=’inner’, on=’email’) # inner оставит только совпадающие значения
pd.merge(table1, table2, how=”left”, on=”email”) # left оставить значения email из левой (table1) и дополнит столбцом из правой таблицы у которых такой же email. Тех email которых нет в левой таблице, не добавяться. how=”right” делает тоже самое, но наоборот)
pd.merge(table1, table2, how=”outer”, on=”email”) # outer оставит все значения email из двух таблиц и отсутсвующие значения в столбцах заполнит NaN
pd.merge(table1, table2, left_index=True,right_on=”email”, how=”inner”) #соединяет таблицы, левую по индексу, а правую по колонке email.
```

Работа со строками
```
names = pd.Series(['andrew','bobo','claire','david','4'])
names.str.isdigit()
names.str.capitalize()
messy_names.str.replace(";","").str.strip().str.capitalize()
```

## Полезные Функции Pandas
Процентное изменение, от предыдущего значения к настоящему
```
df.change = df.price.pct_change()
```

Преобразовать (конвертировать) значения в численный тип:
```
df[['price','qty','quoteQty']] = df[['price','qty','quoteQty']].astype(float)
```

```
Matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axex.plot(x,y)

plt.show()
```

## Seaborn
```
import seaborn as sns
```

График bar plot
```
sns.countplot(data=df, x='target')
```

График pairplot
```
sns.pairplot(df[['age','trestbps', 'chol','thalach','target']], hue='target')
```

График histplot
```
sns.histplot(data=df,x='tenure',bins=60)
```

График countplot
```
sns.countplot(x='type', hue='quality', data=df)
```

Heatmap который показывает корреляцию между колонками
```
plt.figure(figsize=(12,8),dpi=200)
sns.heatmap(df.corr(), annot=True)
```

Тепловая карта для кореляции
```
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),cmap='coolwarm')
```

График кластер мап
```
sns.clustermap(df.corr(),cmap='viridis')
```

Повернуть обозначения оси x на 90 градусов для лучшего отображения
```
plt.xticks(rotation=90);
```

## SKLearn
Разбиение данных на тестовый и обучающий набор
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101) #X - признаки, y - целевая переменная
```

Масштабирование признаков
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

ElasticNet
```
from sklearn.linear_model import ElasticNet
elastic_model = ElasticNet()
```

Или

```
from sklearn.linear_model import ElasticNetCV
elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                          eps=0.001, n_alphas=100,max_iter=1000000)
```

GridSearchCV
```
param_grid = {'alpha':[0.1, 1, 5, 50, 100],
             'l1_ratio':[.1, .5, .7, .95, .99, 1]}

from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(estimator=elastic_model,
                         param_grid=param_grid,
                         scoring='neg_mean_squared_error',
                         cv=5,
                         verbose=2)
grid_model.fit(X_train,y_train)
grid_model.best_params_

y_pred = grid_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_test,y_pred)
np.sqrt(mean_squared_error(y_test,y_pred))
```

Логистическая регрессия мультиклассовая

```
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV #поиск по сетке
clf = LogisticRegressionCV(max_iter=5000).fit(X_train, y_train)

clf.C_ #отобразить параметр С
log_model.get_params() #отобразить другие параметры
clf.coef_ #отобразить коефициенты модели
```

Оценка работы модели
```
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
y_pred = clf.predict(X_test)
confusion_matrix(y_test,y_pred)
plot_confusion_matrix(clf,X_test,y_test)
print(classification_report(y_test,y_pred))
```

KNN Метод ближайших соседей
```
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

"confusion matrix" и отчёт "classification report"
from sklearn.metrics import confusion_matrix,classification_report
grid_pred = grid_model.predict(X_test)
confusion_matrix(y_test,grid_pred)
```

Дерево решений Decision Tree Classificer
```
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train, y_train)
```

Посмотреть важность признаков для дерева решений
```
dt.feature_importances_ 

#или отобразить красиво на графике

imp_feats = pd.DataFrame(data=dt.feature_importances_, index=X.columns,columns=['Важность'])
imp_feats = imp_feats.sort_values('Важность')
imp_feats = imp_feats[imp_feats['Важность'] > 0]
plt.figure(figsize=(10,4),dpi=200)
sns.barplot(data=imp_feats,x=imp_feats.index,y='Важность')
plt.xticks(rotation=90);
```

Случайный лес (Random Forest)
```
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=6)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
```

Расширяемые деревья (Boosted Trees) модель AdaBoost или Gradient Boosting
```
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
ada_model = AdaBoostClassifier(n_estimators=100)
gb_model = GradientBoostingClassifier()

ada_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)


NLP (Natural Language Processing)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
pipe = Pipeline([('tfidf', TfidfVectorizer()),('svc', LinearSVC()),])
pipe.fit(X_train, y_train) 
```

## Pipeline
```
from sklearn.pipeline import Pipeline

operations = [('scaler',scaler),('knn',knn)]
pipe = Pipeline(operations)
```

## DBSCAN

```
from sklearn.cluster import DBSCAN
number_of_outliers = []
for eps in np.linspace(0.001,3,50):
    dbscan = DBSCAN(eps=eps, min_samples=2*df_test.shape[1])
    dbscan.fit_predict(df_test)
    # Сохраняем количество точек выбросов
    number_of_outliers.append(np.sum(dbscan.labels_ == -1))
```

Метод главных компонент (PCA - Principal Component Analysis)
```
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(pixels_scaled)

np.sum(pca.explained_variance_ratio_) #сколько вариативности объясняется этими 2 главными компонентами
```
