from util import argumentos, load_dataset, data_treatment, vectorize_text, mlflow_tracking

def main():
    print("Ejecutamos el main")
    args_values = argumentos()
    df = load_dataset()
    X_train, X_test, y_train, y_test = data_treatment(df)
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_text(X_train, X_test, args_values.max_features)
    mlflow_tracking(args_values.nombre_job, X_train_vectorized, X_test_vectorized, y_train, y_test, args_values.max_iter, args_values.C_values)

if __name__ == "__main__":
    main()