import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
import pickle

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

@app.route('/')
def choose_prediction_method():
    return render_template('index.html')

def upr_prediction(params):
    # загрузка масштабаторов
    scaler_X = pickle.load(open('models/scaler_X_upr.pickle', 'rb'))
    scaler_y = pickle.load(open('models/scaler_y_upr.pickle', 'rb'))

    # загрузка модели
    model = pickle.load(open('models/svr_upr.pickle', 'rb'))

    # масштабирование входных значений
    X_scl = scaler_X.transform(np.array(params).reshape(1, -1))

    # прогноз
    y_pred_scl = model.predict(X_scl)

    # обратное масштабирование прогноза
    y_pred = scaler_y.inverse_transform(y_pred_scl.reshape(-1, 1))

    return y_pred

def pr_prediction(params):

    # загрузка масштабаторов
    scaler_X = pickle.load(open('models/scaler_X_pr.pickle', 'rb'))
    scaler_y = pickle.load(open('models/scaler_y_pr.pickle', 'rb'))

    # загрузка модели
    model = pickle.load(open('models/gb_pr.pickle', 'rb'))

    # масштабирование входных значений
    X_scl = scaler_X.transform(np.array(params).reshape(1, -1))

    # прогноз
    y_pred_scl = model.predict(X_scl)

    # обратное масштабирование прогноза
    y_pred = scaler_y.inverse_transform(y_pred_scl.reshape(-1, 1))

    return y_pred

def mn_prediction(params):

    # загрузка масштабаторов
    scaler_X = pickle.load(open('models/scaler_X_mn.pickle', 'rb'))
    scaler_y = pickle.load(open('models/scaler_y_mn.pickle', 'rb'))

    # загрузка модели
    model = tf.keras.models.load_model('models/best_model_mn.h5')

    # масштабирование входных значений
    X_scl = scaler_X.transform(np.array(params).reshape(1, -1))

    # прогноз
    y_pred_scl = model.predict(X_scl)

    # обратное масштабирование прогноза
    y_pred = scaler_y.inverse_transform(y_pred_scl.reshape(-1, 1))

    return y_pred

@app.route('/upr/', methods=['POST', 'GET'])
def upr_predict():
    message = ''
    invalid_params = []

    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'un', 'ps', 'shn', 'pln')
        params = []

        for i, param_name in enumerate(param_list, start=1):
            param = request.form.get(param_name)

            try:
                param_float = float(param.replace(',', '.'))
            except ValueError:
                invalid_params.append(i)
                continue

            if param_name == 'mn' and not (0 <= param_float <= 6):
                invalid_params.append(i)
            elif param_name == 'plot' and not (1700 <= param_float <= 2300):
                invalid_params.append(i)
            elif param_name == 'mup' and not (2 <= param_float <= 2000):
                invalid_params.append(i)
            elif param_name == 'ko' and not (17 <= param_float <= 200):
                invalid_params.append(i)
            elif param_name == 'seg' and not (14 <= param_float <= 34):
                invalid_params.append(i)
            elif param_name == 'tv' and not (100 <= param_float <= 414):
                invalid_params.append(i)
            elif param_name == 'pp' and not (0.6 <= param_float <= 1400):
                invalid_params.append(i)
            elif param_name == 'un' and param_float not in [0, 90]:
                invalid_params.append(i)
            elif param_name == 'ps' and not (33 <= param_float <= 414):
                invalid_params.append(i)
            elif param_name == 'shn' and not (0 <= param_float <= 15):
                invalid_params.append(i)
            elif param_name == 'pln' and not (0 <= param_float <= 104):
                invalid_params.append(i)

            params.append(param_float)

        if invalid_params:
            error_message = f'Ошибка: Некорректные значения параметра № {", ".join(map(str, invalid_params))}'
            return render_template('upr.html', message=error_message)

        message = f'Спрогнозированное значение прочности при растяжении для введённых параметров: {upr_prediction(params)[0, 0]} ГПа'

    return render_template('upr.html', message=message)

@app.route('/pr/', methods=['POST', 'GET'])
def pr_predict():
    message = ''
    invalid_params = []

    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'un', 'ps', 'shn', 'pln')
        params = []

        for i, param_name in enumerate(param_list, start=1):
            param = request.form.get(param_name)

            try:
                param_float = float(param.replace(',', '.'))
            except ValueError:
                invalid_params.append(i)
                continue

            if param_name == 'mn' and not (0 <= param_float <= 6):
                invalid_params.append(i)
            elif param_name == 'plot' and not (1700 <= param_float <= 2300):
                invalid_params.append(i)
            elif param_name == 'mup' and not (2 <= param_float <= 2000):
                invalid_params.append(i)
            elif param_name == 'ko' and not (17 <= param_float <= 200):
                invalid_params.append(i)
            elif param_name == 'seg' and not (14 <= param_float <= 34):
                invalid_params.append(i)
            elif param_name == 'tv' and not (100 <= param_float <= 414):
                invalid_params.append(i)
            elif param_name == 'pp' and not (0.6 <= param_float <= 1400):
                invalid_params.append(i)
            elif param_name == 'un' and param_float not in [0, 90]:
                invalid_params.append(i)
            elif param_name == 'ps' and not (33 <= param_float <= 414):
                invalid_params.append(i)
            elif param_name == 'shn' and not (0 <= param_float <= 15):
                invalid_params.append(i)
            elif param_name == 'pln' and not (0 <= param_float <= 104):
                invalid_params.append(i)

            params.append(param_float)

        if invalid_params:
            error_message = f'Ошибка: Некорректные значения параметра № {", ".join(map(str, invalid_params))}'
            return render_template('pr.html', message=error_message)

        message = f'Спрогнозированное значение прочности при растяжении для введённых параметров: {pr_prediction(params)[0, 0]} ГПа'

    return render_template('pr.html', message=message)



@app.route('/mn/', methods=['POST', 'GET'])
def mn_predict():
    message = ''
    invalid_params = []

    if request.method == 'POST':
        param_list = ('upr', 'pr', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'un', 'ps', 'shn', 'pln')
        params = []

        for i, param_name in enumerate(param_list, start=1):
            param = request.form.get(param_name)

            try:
                param_float = float(param.replace(',', '.'))
            except ValueError:
                invalid_params.append(i)
                continue

            if param_name == 'pr' and not (1250 <= param_float <= 3705):
                invalid_params.append(i)
            elif param_name == 'upr' and not (65 <= param_float <= 81):
                invalid_params.append(i)
            elif param_name == 'plot' and not (1700 <= param_float <= 2300):
                invalid_params.append(i)
            elif param_name == 'mup' and not (2 <= param_float <= 2000):
                invalid_params.append(i)
            elif param_name == 'ko' and not (17 <= param_float <= 200):
                invalid_params.append(i)
            elif param_name == 'seg' and not (14 <= param_float <= 34):
                invalid_params.append(i)
            elif param_name == 'tv' and not (100 <= param_float <= 414):
                invalid_params.append(i)
            elif param_name == 'pp' and not (0.6 <= param_float <= 1400):
                invalid_params.append(i)
            elif param_name == 'un' and param_float not in [0, 90]:
                invalid_params.append(i)
            elif param_name == 'ps' and not (33 <= param_float <= 414):
                invalid_params.append(i)
            elif param_name == 'shn' and not (0 <= param_float <= 15):
                invalid_params.append(i)
            elif param_name == 'pln' and not (0 <= param_float <= 104):
                invalid_params.append(i)

            params.append(param_float)

        if invalid_params:
            error_message = f'Ошибка: Некорректные значения параметра № {", ".join(map(str, invalid_params))}'
            return render_template('mn.html', message=error_message)

        message = f'Спрогнозированное значение соотношения матрица-наполнитель для введённых параметров: {mn_prediction(params)[0, 0]} ГПа'

    return render_template('mn.html', message=message)

if __name__ == '__main__':
    app.debug = True
    app.run()
