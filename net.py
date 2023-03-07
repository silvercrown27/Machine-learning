import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import keras
from keras import layers
from keras import callbacks

pd.plotting.register_matplotlib_converters()

plt.style.use('seaborn-whitegrid')

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize='large', titlepad=10)
plt.rc('animation', html='html5')


path = f"datasets/home-data-for-ml-course/train.csv"
data = pd.read_csv(path)
df = data.copy()

print(df.head())
print(df.columns)

cat_cols = [col for col in df.columns if df[col].dtype == 'object']

label_df = df.copy()

encoder = OrdinalEncoder()
label_df[cat_cols] = encoder.fit_transform(df[cat_cols])

print(label_df[:])


X = label_df.copy()

train = X.sample(frac=0.75, random_state=1)
valid = X.drop(train.index)

y_train = train.pop('SalePrice')
y_valid = valid.pop('SalePrice')

max_ = train.max(axis=0)
min_ = train.min(axis=0)

train = (train - min_) / (max_ - min_)
valid = (valid - min_) / (max_ - min_)


impt = SimpleImputer()

X_train = pd.DataFrame(impt.fit_transform(train))
X_valid = pd.DataFrame(impt.transform(valid))

X_train.columns = train.columns
X_valid.columns = valid.columns

input_shape = [X_train.shape[1]]

model = keras.Sequential([
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1024, input_shape=input_shape, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1),
])

model.compile(loss='mae', optimizer='adam')

earlystopping = callbacks.EarlyStopping(min_delta=0.001, patience=50, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=512, epochs=5000, callbacks=[earlystopping])

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss']].plot()

print(history_df.head())

plt.show()