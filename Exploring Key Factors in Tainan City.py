#!/usr/bin/env python
# coding: utf-8

import pandas as pd

df1 = pd.read_excel("C:/Users/s0958/Downloads/tainan/1130106tainan.xlsx")
df2 = pd.read_excel("C:/Users/s0958/Downloads/tainan/1130712tainan.xlsx")

print(df1.columns.equals(df2.columns)) 

df = pd.concat([df1, df2], ignore_index=True)

print(df.shape)
df.head()


df.columns



#missing value
print("\n遺失值數量:")
print(df.isnull().sum()) 


categorical_cols = df.select_dtypes(include=['object']).columns
categorical_summary = {}

excluded_columns = ['總編號', '案件類別']
categorical_cols = [col for col in categorical_cols if col not in excluded_columns]
    
for column in categorical_cols:
        categorical_summary[column] = df[column].value_counts().head(10) 

for column, freq in categorical_summary.items():
        print(f"\n{column}的敘述統計總覽:")
        print(freq)


df["是否死亡"] = ((df["24小時內死亡人數"].fillna(0) + df["30日內死亡人數"].fillna(0)) > 0).astype(int)

df_death = df[df["是否死亡"] == 1].copy()
df_death["是否為A1事故"] = (df_death["24小時內死亡人數"].fillna(0) > 0).astype(int)

# A1 vs not A1
print(df_death.shape)
df_death


print(df_death["是否為A1事故"].value_counts())


categorical_cols = df_death.select_dtypes(include=['object']).columns
categorical_summary = {}

excluded_columns = ['總編號', '案件類別']
categorical_cols = [col for col in categorical_cols if col not in excluded_columns]

for column in categorical_cols:
    value_counts = df_death[column].value_counts()
    total = value_counts.sum()
    formatted = value_counts.apply(lambda x: f"{x} ({x / total:.1%})")
    categorical_summary[column] = formatted


for column, freq in categorical_summary.items():
    print(f"\n{column}的敘述統計總覽:")
    print(freq)




print(df_death["是否為A1事故"].value_counts())
import matplotlib.pyplot as plt

value_counts = df_death["是否為A1事故"].value_counts(normalize=True) * 100

plt.figure(figsize=(6, 3))
bars = plt.barh(value_counts.index.astype(str), value_counts.values, color='skyblue')

for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

plt.title("是否為A1事故 - 百分比分布")
plt.xlabel("百分比 (%)")
plt.ylabel("是否為A1事故")
plt.xlim(0, max(value_counts.values) + 10)  # 預留空間給文字
plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt

all_columns = categorical_cols + ["是否為A1事故"]

fig, axes = plt.subplots(nrows=len(all_columns), ncols=1, figsize=(16, len(all_columns) * 6))

for i, column in enumerate(all_columns):
    ax = axes[i]
    value_counts = df_death[column].value_counts(normalize=True) * 100  # 顯示完整類別百分比

    bars = ax.barh(value_counts.index.astype(str), value_counts.values, color='skyblue')
    ax.set_title(f"{column} 類別統計（完整類別）")
    ax.set_xlabel("百分比 (%)")
    ax.invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va='center')

plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt


all_columns = categorical_cols + ["是否為A1事故"]

for column in all_columns:
    value_counts = df_death[column].value_counts(normalize=True).head(5) * 100  # 取前五大類別


    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.barh(value_counts.index.astype(str), value_counts.values, color='skyblue')
    ax.set_title(f"{column} 前五大類別統計")
    ax.set_xlabel("百分比 (%)")
    ax.invert_yaxis()


    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va='center')

    plt.tight_layout()
    plt.show()




from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score





features = ['天候', '道路照明設備', '道路型態', '事故位置', '號誌-號誌種類', '事故類型及型態',"初步分析研判子類別-主要"]
target = '是否為A1事故'

X = df_death[features]
y = df_death[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  
X_train_transformed = encoder.fit_transform(X_train)
X_test_transformed = encoder.transform(X_test)




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
plt.rcParams['font.family'] = 'Microsoft YaHei'
#  Cramér's V
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

cramers_v_matrix = pd.DataFrame(np.zeros((X.shape[1], X.shape[1])), columns=X.columns, index=X.columns)

for col1 in X.columns:
    for col2 in X.columns:
        cramers_v_matrix.loc[col1, col2] = cramers_v(X[col1], X[col2])


plt.figure(figsize=(10, 8))
sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=0, vmax=1)
plt.title('Cramér\'s V Heatmap of Categorical Features')
plt.show()



features = ['天候', '道路照明設備', '事故位置', '號誌-號誌種類', '事故類型及型態']
target = '是否為A1事故'

X = df_death[features]
y = df_death[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot 
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  
X_train_transformed = encoder.fit_transform(X_train)
X_test_transformed = encoder.transform(X_test)

# Cramér's V
cramers_v_matrix = pd.DataFrame(np.zeros((X.shape[1], X.shape[1])), columns=X.columns, index=X.columns)

for col1 in X.columns:
    for col2 in X.columns:
        cramers_v_matrix.loc[col1, col2] = cramers_v(X[col1], X[col2])


plt.figure(figsize=(10, 8))
sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=0, vmax=1)
plt.title('Cramér\'s V Heatmap of Categorical Features')
plt.show()


model = LogisticRegression(max_iter=1000)
model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_test_transformed)

print("準確率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))





feature_names = encoder.get_feature_names_out(features)
coefficients = model.coef_[0]

feature_importance = pd.Series(coefficients, index=feature_names).sort_values(ascending=False)

print("影響最大的正向因子：")
print(feature_importance.head())

print("\n影響最大的負向因子：")
print(feature_importance.tail())




import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

plt.rcParams['font.family'] = 'Microsoft YaHei'

feature_names = encoder.get_feature_names_out(features)
coefficients = model.coef_[0]

sorted_indices = np.argsort(coefficients)
sorted_features = feature_names[sorted_indices]
sorted_coefficients = coefficients[sorted_indices]

plt.figure(figsize=(14, 16))
plt.barh(sorted_features, sorted_coefficients, color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Coefficients')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

# ROC 
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.grid()
plt.show()




from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
ap_score = average_precision_score(y_test, y_pred_proba)

print(f"Average Precision (AUPRC): {ap_score:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AUPRC = {ap_score:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()





from sklearn.metrics import confusion_matrix
import seaborn as sns


cm = confusion_matrix(y_test, model.predict(X_test_transformed))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: 0', 'Pred: 1'], yticklabels=['True: 0', 'True: 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()






coefficients = model.coef_[0]


intercept = model.intercept_[0]


equation = f"Logistic Regression: log(p/(1-p)) = {intercept:.4f} + " + " + ".join([f"{coeff:.4f}*X{i}" for i, coeff in enumerate(coefficients)])
print(equation)


terms = [f"{coef:+.4f}*{name}" for coef, name in zip(coefficients, feature_names)]

formula = f"log(p / (1 - p)) = {intercept:+.4f} " + " ".join(terms)
print(formula)

formatted_coeffs = [f"{coef:.4f}" for coef in coefficients]
print(" ".join(formatted_coeffs))





def compute_aic(model, X, y):
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred_prob = np.clip(y_pred_prob, 1e-10, 1 - 1e-10)  # 避免 log(0)
    log_likelihood = np.sum(y * np.log(y_pred_prob) + (1 - y) * np.log(1 - y_pred_prob))
    k = X.shape[1] + 1  # 參數數量（含截距）
    AIC = 2 * k - 2 * log_likelihood
    return AIC

def compute_bic(model, X, y):
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred_prob = np.clip(y_pred_prob, 1e-10, 1 - 1e-10)
    log_likelihood = np.sum(y * np.log(y_pred_prob) + (1 - y) * np.log(1 - y_pred_prob))
    k = X.shape[1] + 1
    n = len(y)
    BIC = np.log(n) * k - 2 * log_likelihood
    return BIC

aic = compute_aic(model, X_train_transformed, y_train)
bic = compute_bic(model, X_train_transformed, y_train)

print(f"AIC: {aic:.2f}")
print(f"BIC: {bic:.2f}")




encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

feature_names = encoder.get_feature_names_out(features)
X_train_df = pd.DataFrame(X_train_encoded, columns=feature_names)
X_test_df = pd.DataFrame(X_test_encoded, columns=feature_names)

from sklearn.metrics import f1_score, balanced_accuracy_score


def backward_elimination_with_metric(X_train, y_train, X_test, y_test,
                                     min_features=5,
                                     metric='f1',
                                     criterion='aic',  # 'aic' 或 'bic'
                                     weight_criterion=0.7,
                                     weight_metric=0.3):
    features = list(X_train.columns)
    best_score = np.inf
    best_model = None
    best_features = features.copy()

    while len(features) > min_features:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train[features], y_train)

        # AIC or BIC
        if criterion == 'aic':
            criterion_value = compute_aic(model, X_train[features], y_train)
        elif criterion == 'bic':
            criterion_value = compute_bic(model, X_train[features], y_train)
        else:
            raise ValueError("criterion 必須是 'aic' 或 'bic'")

      
        y_pred = model.predict(X_test[features])
        if metric == 'f1':
            score_metric = f1_score(y_test, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            score_metric = balanced_accuracy_score(y_test, y_pred)
        else:
            raise ValueError("metric 必須是 'f1' 或 'balanced_accuracy'")

       
        normalized_crit = criterion_value / 1000
        score = weight_criterion * normalized_crit - weight_metric * score_metric

        if score < best_score:
            best_score = score
            best_model = model
            best_features = features.copy()

   
        coef_abs = np.abs(model.coef_[0])
        min_coef_idx = np.argmin(coef_abs)
        features.pop(min_coef_idx)

    return best_model, best_features, best_score

criterion = 'aic' 

final_model, selected, score = backward_elimination_with_metric(
    X_train_df, y_train, X_test_df, y_test,
    min_features=5,
    metric='balanced_accuracy',
    criterion=criterion,
    weight_criterion=0.7,
    weight_metric=0.3
)

print(f"保留特徵數: {len(selected)}")
print("保留的特徵:")
for f in selected:
    print(f)

final_aic = compute_aic(final_model, X_train_df[selected], y_train)
final_bic = compute_bic(final_model, X_train_df[selected], y_train)

print(f"\n最佳 AIC: {final_aic:.2f}")
print(f"最佳 BIC: {final_bic:.2f}")



y_pred_final = final_model.predict(X_test_df[selected])
print("準確率:", accuracy_score(y_test, y_pred_final))
print("\n分類報告：")
print(classification_report(y_test, y_pred_final))






coefficients = final_model.coef_[0]


intercept = final_model.intercept_[0]


equation = f"Logistic Regression: log(p/(1-p)) = {intercept:.4f} + " + " + ".join([f"{coeff:.4f}*X{i}" for i, coeff in enumerate(coefficients)])
print(equation)


terms = [f"{coef:+.4f}*{name}" for coef, name in zip(coefficients, X_test_df[selected])]

formula = f"log(p / (1 - p)) = {intercept:+.4f} " + " ".join(terms)
print(formula)

formatted_coeffs1 = [f"{coef:.4f}" for coef in coefficients]
print(" ".join(formatted_coeffs1))





from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_pred_proba2 = final_model.predict_proba(X_test_df[selected])[:, 1]


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba2)

# AUC 
auc_score = roc_auc_score(y_test, y_pred_proba2)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # 隨機分類器基準線
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()






precision2, recall2, _ = precision_recall_curve(y_test, y_pred_proba2)
ap_score2 = average_precision_score(y_test, y_pred_proba2)

print(f"Average Precision (AUPRC): {ap_score2:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall2, precision2, color='darkorange', lw=2, label=f'Precision-Recall curve (AUPRC = {ap_score2:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()





import seaborn as sns

coef = final_model.coef_[0]
importance = pd.Series(coef, index=selected)


plt.figure(figsize=(10, 8))
importance.sort_values(ascending=True).plot(kind='barh')
plt.title("Feature Importance (Coefficient Magnitude)")
plt.xlabel("Coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()





importance




from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


precision1, recall1, _ = precision_recall_curve(y_test, y_pred_proba)
ap_score1 = average_precision_score(y_test, y_pred_proba)


precision2, recall2, _ = precision_recall_curve(y_test, y_pred_proba2)
ap_score2 = average_precision_score(y_test, y_pred_proba2)

plt.figure(figsize=(8, 6))
plt.plot(recall1, precision1, label=f'全模型 (AUPRC = {ap_score1:.4f})', color='darkorange', lw=2)
plt.plot(recall2, precision2, label=f'簡化模型 (AUPRC = {ap_score2:.4f})', color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()






fpr1, tpr1, _ = roc_curve(y_test, y_pred_proba)
fpr2, tpr2, _ = roc_curve(y_test, y_pred_proba2)
auc1 = roc_auc_score(y_test, y_pred_proba)
auc2 = roc_auc_score(y_test, y_pred_proba2)


plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f'全模型 (AUC = {auc1:.2f})', color='darkorange')
plt.plot(fpr2, tpr2, label=f'簡化模型 (AUC = {auc2:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='隨機分類基準')

plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve 比較')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()




import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

results = []


criteria = ['aic', 'bic']
metrics = ['f1', 'balanced_accuracy']
weights = [round(w, 2) for w in np.linspace(0, 1, 21)]  # 0.00 ~ 1.00 間隔 0.05

for criterion in tqdm(criteria, desc="Criterion"):
    for metric in metrics:
        for w_crit in weights:
            w_metric = 1 - w_crit
            try:
                model, selected, score = backward_elimination_with_metric(
                    X_train_df, y_train, X_test_df, y_test,
                    min_features=5,
                    metric=metric,
                    criterion=criterion,
                    weight_criterion=w_crit,
                    weight_metric=w_metric
                )


                y_pred = model.predict(X_test_df[selected])
                y_pred_proba = model.predict_proba(X_test_df[selected])[:, 1]

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                bacc = balanced_accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                auprc = average_precision_score(y_test, y_pred_proba)
                aic = compute_aic(model, X_train_df[selected], y_train)
                bic = compute_bic(model, X_train_df[selected], y_train)

                results.append({
                    'criterion': criterion,
                    'metric': metric,
                    'weight_c': w_crit,
                    'weight_m': w_metric,
                    'accuracy': acc,
                    'f1': f1,
                    'balanced_accuracy': bacc,
                    'roc_auc': roc_auc,
                    'auprc': auprc,
                    'aic': aic,
                    'bic': bic,
                    'selected_features': selected,
                    'num_features': len(selected),
                    'composite_score': score
                })
            except Exception as e:
                print(f"組合失敗：{criterion}, {metric}, w_crit={w_crit}，錯誤訊息：{e}")


results_df = pd.DataFrame(results)





results_df





best_aic = results_df.loc[results_df['aic'].idxmin()]
best_bic = results_df.loc[results_df['bic'].idxmin()]
best_acc = results_df.loc[results_df['accuracy'].idxmax()]
best_f1 = results_df.loc[results_df['f1'].idxmax()]
best_bacc = results_df.loc[results_df['balanced_accuracy'].idxmax()]
best_roc = results_df.loc[results_df['roc_auc'].idxmax()]
best_auprc = results_df.loc[results_df['auprc'].idxmax()]
best_composite = results_df.loc[results_df['composite_score'].idxmin()]  # 最小的總分


summary_df = pd.DataFrame([
    best_aic, best_bic, best_acc, best_f1, best_bacc, best_roc, best_auprc, best_composite
], index=[
    'AIC 最小', 'BIC 最小', '準確率最高', 'F1 分數最高', '平衡準確率最高', 'ROC AUC 最高', 'AUPRC 最高', '綜合評分最佳'
])

summary_df = summary_df[['criterion', 'metric', 'weight_c', 'weight_m',
                         'accuracy', 'f1', 'balanced_accuracy', 'roc_auc', 'auprc',
                         'aic', 'bic', 'num_features']]

summary_df





results_df.to_excel("results.xlsx", index=False)






