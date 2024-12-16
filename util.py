import argparse
import subprocess
import time
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def argumentos():
    parser = argparse.ArgumentParser(description='__main__ de la aplicación con argumentos de entrada.')
    parser.add_argument('--nombre_job', type=str, help='Valor para el parámetro nombre_documento.')
    parser.add_argument('--max_features', type=int, default=8000, help='Número máximo de características para TfidfVectorizer.')
    parser.add_argument('--max_iter', type=int, default=1000, help='Número máximo de iteraciones para LogisticRegression.')
    parser.add_argument('--C_values', nargs='+', type=float, default=[0.1, 1.0, 10.0], help='Valores de C para LogisticRegression.')
    return parser.parse_args()

def load_dataset():
    df_subset = pd.read_csv('datos_procesados.csv')
    df_subset['label'] = (df_subset['rating'] >= 4).astype(int)
    return df_subset

def data_treatment(df):
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test, max_features):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, vectorizer

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report, confusion_matrix(y_test, y_pred)

def mlflow_tracking(nombre_job, X_train, X_test, y_train, y_test, max_iter, C_values):
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui', '--port', '8888'])
    print(mlflow_ui_process)
    time.sleep(5)
    mlflow.set_experiment(nombre_job)
    
    for C in C_values:
        with mlflow.start_run(run_name=f"LogisticRegression_C_{C}_MaxIter_{max_iter}"):
            model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
            model.fit(X_train, y_train)            

            # Calcular accuracy de entrenamiento y prueba
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)

            # Evaluar el modelo
            classification_rep, conf_matrix = evaluate_model(model, X_test, y_test)
            
            # Logging de métricas y parámetros
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("accuracy", classification_rep['accuracy'])
            mlflow.log_metric("precision", classification_rep['weighted avg']['precision'])
            mlflow.log_metric("recall", classification_rep['weighted avg']['recall'])
            mlflow.log_metric("f1-score", classification_rep['weighted avg']['f1-score'])
            mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")
            
            # Guardar el modelo
            mlflow.sklearn.log_model(model, "logistic_regression_model")
    
    print("Se ha acabado el entrenamiento de los modelos correctamente")