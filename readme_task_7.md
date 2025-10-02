# Task 7: Support Vector Machines (SVM) (AI & ML Internship)

## 🎯 Objective
Implement Support Vector Machines (SVM) for binary classification using the Breast Cancer dataset and evaluate performance using different kernels.

## 📂 Dataset
- **Breast Cancer Dataset** (`breast-cancer.csv`)
- Target variable: `diagnosis` (Malignant = 1, Benign = 0)
- Features: 30 numeric attributes related to cell nuclei measurements

## 🛠️ Steps Performed
1. **Imported dataset**
   - Loaded `breast-cancer.csv` using Pandas
   - Dropped irrelevant columns (`id`, `Unnamed: 32`)

2. **Prepared features and target variable**
   - `X = df.drop('diagnosis', axis=1)`
   - `y = df['diagnosis'].map({'M':1, 'B':0})`

3. **Train/test split and standardization**
   - Split dataset into 80% training and 20% testing
   - Applied `StandardScaler` to normalize features

4. **Trained Linear SVM**
   - Built model with kernel = `linear`
   - Evaluated accuracy, classification report, and confusion matrix

5. **Trained RBF Kernel SVM**
   - Built model with kernel = `rbf`
   - Evaluated accuracy, classification report, and confusion matrix

6. **Hyperparameter tuning with GridSearchCV**
   - Tuned `C` and `gamma` for RBF kernel
   - Selected best parameters and evaluated accuracy

7. **Decision boundary visualization (PCA)**
   - Reduced dataset to 2D using PCA
   - Plotted SVM decision boundaries for visualization

## 📊 Evaluation
- **Linear SVM Accuracy:** Good baseline performance
- **RBF SVM Accuracy:** Higher accuracy due to capturing non-linear boundaries
- **GridSearchCV:** Helped in selecting best `C` and `gamma`
- **Confusion Matrix:** Showed classification strengths and misclassifications
- **Decision Boundary:** Visualized separation between classes after PCA

## 🧰 Tools Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## 📦 Deliverables
- Jupyter Notebook: `task7.ipynb`
- Breast Cancer Dataset: `breast-cancer.csv`
- README documentation (this file)


