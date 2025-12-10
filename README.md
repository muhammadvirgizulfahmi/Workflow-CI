# Submission MLOps: Student Placement Prediction Pipeline

![CI/CD Status](https://github.com/muhammadvirgizulfahmi/Workflow-CI/actions/workflows/main.yml/badge.svg)

Repository ini berisi implementasi **End-to-End MLOps Pipeline** untuk memprediksi penempatan kerja mahasiswa (*Student Placement*) menggunakan algoritma **Random Forest**. Project ini mencakup pelatihan model otomatis, pelacakan eksperimen dengan MLflow, dan deployment otomatis menggunakan Docker melalui GitHub Actions.

## Struktur Project

Workflow-CI/
├── .github/workflows/
│   └── main.yml           # Konfigurasi CI/CD Pipeline
├── MLProject/
│   ├── MLproject          # File definisi MLflow Project
│   ├── conda.yaml         # Environment dependencies
│   ├── modelling.py       # Script training & evaluasi model
│   └── college_student_placement_dataset_preprocessing.csv # Dataset
└── README.md