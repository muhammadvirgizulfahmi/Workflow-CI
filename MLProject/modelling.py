import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
# import dagshub = tidak dipakai
import os

# # Setup Dagshub
# dagshub.init(repo_owner='muhammadvirgizulfahmi', 
#              repo_name='Eksperimen_Model_Muhammad-Virgi-Zulfahmi', 
#              mlflow=True)

# mlflow.set_experiment("Eksperimen_Membangun_Model")

def train_model():
    # Memuat data
    try:
        df = pd.read_csv("college_student_placement_dataset_preprocessing.csv")
    except FileNotFoundError:
        print("Error: File csv tidak ditemukan!")
        return
    
    X = df.drop(columns=['Placement'])
    y = df['Placement']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Modelling"):
        # Inisialisasi & Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAkurasi Model: {acc:.4f}")

        ## Membuat artefak

        # ARTEFAK 1: Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Placed', 'Placed'], yticklabels=['Not Placed', 'Placed'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # ARTEFAK 2: Feature Importance Plot
        plt.figure(figsize=(10, 6))
        feature_names = X.columns
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
        plt.savefig("feature_importance.png")
        plt.close()
        mlflow.log_artifact("feature_importance.png")
        
        print("\nTraining selesai.\n")

        # Bersihkan file lokal
        if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")
        if os.path.exists("feature_importance.png"): os.remove("feature_importance.png")

if __name__ == "__main__":
    train_model()