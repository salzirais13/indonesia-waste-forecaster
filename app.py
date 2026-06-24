from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the data and model once at the start
data = pd.read_csv('hotdeckcsv.csv', sep=";")
grouped_data = data.groupby(['Prov', 'Tahun'], as_index=False).sum()
scaler = MinMaxScaler()
grouped_data['TS'] = scaler.fit_transform(grouped_data[['TS']])
data_pivot = grouped_data.pivot_table(index=['Prov', 'Tahun'], values='TS').reset_index()
prov_data = {prov: data_pivot[data_pivot['Prov'] == prov].drop(['Prov'], axis=1) for prov in data_pivot['Prov'].unique()}

look_back = 3

def create_dataset(df, look_back=1):
    data = df.values
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

X_all, y_all = [], []

for prov, df in prov_data.items():
    X, y = create_dataset(df, look_back)
    X_all.append(X)
    y_all.append(y)

X_all = np.vstack(X_all)
y_all = np.hstack(y_all)
split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

model = Sequential()
model.add(SimpleRNN(units=50, activation='relu', input_shape=(look_back, 1),
                    kernel_initializer=tf.constant_initializer(0.1),
                    recurrent_initializer=tf.constant_initializer(0.1),
                    bias_initializer=tf.constant_initializer(0.1)))
model.add(Dense(1, kernel_initializer=tf.constant_initializer(0.1), bias_initializer=tf.constant_initializer(0.1)))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(X_train, y_train, epochs=300, batch_size=16, validation_split=0.1, shuffle=False)

def predict_future_values(model, data, look_back, num_predictions):
    predictions = []
    current_data = data[-look_back:]
    for _ in range(num_predictions):
        current_data_reshaped = np.reshape(current_data, (1, look_back, 1))
        next_value = model.predict(current_data_reshaped)[0, 0]
        predictions.append(next_value)
        current_data = np.append(current_data, next_value)[-look_back:]
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    provinsi = None
    tahun = None
    if request.method == 'POST':
        target_provinsi = request.form['provinsi']
        tahun_akhir = int(request.form['tahun'])
        
        if target_provinsi not in prov_data:
            result = {
                'error': f"Province '{target_provinsi}' not found in data."
            }
        else:
            provinsi = target_provinsi
            tahun = tahun_akhir
            
            df = prov_data[target_provinsi]
            last_values = df['TS'].values[-look_back:]

            target_tahun = list(range(2024, tahun_akhir + 1))
            preds = predict_future_values(model, last_values, look_back, len(target_tahun))

            future_df = pd.DataFrame({'Prov': [target_provinsi] * len(target_tahun), 'Tahun': target_tahun, 'TS': preds})
            future_df['TS'] = scaler.inverse_transform(future_df[['TS']])
            
            last_ts = future_df['TS'].iloc[-1]
            tpa = pd.read_csv('tpa.csv', sep=";")
            kapasitas_2 = tpa.loc[tpa['Provinsi'] == target_provinsi, 'Kapasitas 2'].values[0]
            persentase = ((last_ts / 365) / kapasitas_2) * 100
            
            result = {
                'last_ts': last_ts,
                'persentase': persentase,
                'provinsi': provinsi,
                'tahun': tahun
            }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
