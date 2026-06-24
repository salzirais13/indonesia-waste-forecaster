
import pandas as pd
import numpy as np
np.random.seed(42)

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/Book1.csv',sep=';')
def hot_deck_imputation(data, column):
    missing_rows = data[data[column].isnull()]
    for index, row in missing_rows.iterrows():
        most_similar_row = data[data[column].notnull()].sort_values(by=column, ascending=False).iloc[0]

        data.loc[index, column] = most_similar_row[column]

    return data
# Menggunakan fungsi hot_deck_imputation untuk mengisi nilai kosong pada kolom 'TS'
data = hot_deck_imputation(data, 'TS')
# Menyimpan data panel ke dalam file CSV
data.to_csv('/content/drive/My Drive/hotdeckcsv.csv', index=False)

data

import pandas as pd
np.random.seed(42)

data = data

# Drop the columns 'Kab/Kot' and 'Code'
data = data.drop(columns=['Kab/Kot'])

# Group by 'Prov' and 'Tahun' and sum the 'TS' column
grouped_data = data.groupby(['Prov', 'Tahun'], as_index=False).sum()

grouped_data.head()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from google.colab import drive
from keras.callbacks import EarlyStopping
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the data
data=grouped_data

# Sort data by province and year
data = data.sort_values(by=['Prov', 'Tahun'])

# Normalize data
scaler = MinMaxScaler()
data['TS'] = scaler.fit_transform(data[['TS']])

# Use Pivot Table to get data in time series format
data_pivot = data.pivot_table(index=['Prov', 'Tahun'], values='TS').reset_index()

# Create data for each province
prov_data = {}
for prov in data_pivot['Prov'].unique():
    prov_df = data_pivot[data_pivot['Prov'] == prov].drop(['Prov'], axis=1)
    prov_data[prov] = prov_df

# Function to convert data to suitable format for RNN
def create_dataset(df, look_back=1):
    data = df.values
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 3  # Number of timesteps to use
X_all, y_all = [], []

for prov, df in prov_data.items():
    X, y = create_dataset(df, look_back)
    X_all.append(X)
    y_all.append(y)

X_all = np.vstack(X_all)
y_all = np.hstack(y_all)

# Split data into training and testing sets
split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# Reshape data for RNN
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=50, activation='relu', input_shape=(look_back, 1),
                    kernel_initializer=tf.constant_initializer(0.1),
                    recurrent_initializer=tf.constant_initializer(0.1),
                    bias_initializer=tf.constant_initializer(0.1)))
model.add(Dense(1, kernel_initializer=tf.constant_initializer(0.1), bias_initializer=tf.constant_initializer(0.1)))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Use EarlyStopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model with more epochs and smaller batch size without shuffling
history = model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.1, callbacks=[early_stop], shuffle=False)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

from sklearn.metrics import mean_absolute_percentage_error

# Hitung MAPE
mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # hasil dalam persen
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

def predict_future_values(model, data, look_back, num_predictions):
    predictions = []
    current_data = data[-look_back:]
    for _ in range(num_predictions):
        current_data_reshaped = np.reshape(current_data, (1, look_back, 1))
        next_value = model.predict(current_data_reshaped)[0, 0]
        predictions.append(next_value)
        current_data = np.append(current_data, next_value)[-look_back:]
    return predictions

# Tentukan provinsi dan tahun akhir yang ingin diprediksi
target_provinsi = 'Aceh'  # Ganti dengan nama provinsi yang diinginkan
tahun_akhir = 2026  # Ganti dengan tahun akhir yang diinginkan

# Ambil data untuk provinsi yang dipilih
df = prov_data[target_provinsi]
last_values = df['TS'].values[-look_back:]

# Looping dari tahun 2024 hingga tahun akhir
target_tahun = list(range(2024, tahun_akhir + 1))

# Prediksi untuk tahun-tahun yang ditentukan
preds = predict_future_values(model, last_values, look_back, len(target_tahun))

# Buat DataFrame hasil prediksi
future_df = pd.DataFrame({'Prov': [target_provinsi] * len(target_tahun), 'Tahun': target_tahun, 'TS': preds})

# Inverse transform hasil prediksi
future_df['TS'] = scaler.inverse_transform(future_df[['TS']])

# Tampilkan DataFrame prediksi
print(future_df)

# Tampilkan hasil TS terakhir
last_ts = future_df['TS'].iloc[-1]
print(f'Hasil output TS terakhir untuk tahun {tahun_akhir}: {last_ts}')

tpa = pd.read_excel('/content/drive/My Drive/tpa.xlsx')
# Cari nilai 'Kapasitas 2' yang sesuai dengan target_provinsi
kapasitas_2 = tpa.loc[tpa['Provinsi'] == target_provinsi, 'Kapasitas 2'].values[0]

# Hitung persentase
persentase = ((last_ts/365) / kapasitas_2) * 100

# Tampilkan hasil persentase
print(f'Persentase untuk {target_provinsi}: {persentase:.2f}%')
