
import joblib
import numpy as np
from configurations import config


class classPrediction:
    def __init__(self) -> None:
        pass

    def pred_inf(self, df_test):

        # Cargar el Modelo ML o Cargar el Pipeline
        pod_pipeline = joblib.load('src/precio_coches_pipeline.joblib')

        # Quitamos etiquetas km y miles y convertimos a numerico
        df_test["running"] = df_test["running"].astype(str)
        df_test["running"] = df_test["running"].str.replace(" km", "")
        df_test["running"] = df_test["running"].str.replace(" miles", "")
        df_test["running"] = df_test["running"].astype(float)
        df_test["running"]

        df_test = df_test[config.FEATURES]

        # Predicciones con pipeline generado
        predicciones = pod_pipeline.predict(df_test)

        # Revirtiendo escalamiento
        predicciones_sin_escalar = np.exp(predicciones)
        return predicciones, predicciones_sin_escalar, df_test

    def contac_prediccion(self,
                          datos_procesados,
                          prediccion,
                          prediccion_sin_escalar):
        df_resultado = datos_procesados.copy()
        df_resultado['Predicción Escalada'] = prediccion
        df_resultado['Predicción Sin Escalar'] = prediccion_sin_escalar

        # Creamos el archivo CSV para descargar
        csv = df_resultado.to_csv(index=False).encode('utf-8')

        return df_resultado, csv
