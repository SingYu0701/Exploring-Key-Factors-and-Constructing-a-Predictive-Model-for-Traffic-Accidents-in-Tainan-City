#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df1 = pd.read_excel("C:/Users/s0958/Downloads/tainan/1130106tainan.xlsx")
df2 = pd.read_excel("C:/Users/s0958/Downloads/tainan/1130712tainan.xlsx")

# 檢查欄位是否一致
print(df1.columns.equals(df2.columns)) 

# 合併資料（垂直串接）
df = pd.concat([df1, df2], ignore_index=True)

# 檢查資料長相
print(df.shape)
df.head()


# In[2]:


df.columns


# In[3]:


#檢查遺失值
print("\n遺失值數量:")
print(df.isnull().sum()) 


# In[4]:


categorical_cols = df.select_dtypes(include=['object']).columns
categorical_summary = {}

excluded_columns = ['總編號', '案件類別']
categorical_cols = [col for col in categorical_cols if col not in excluded_columns]
    
for column in categorical_cols:
        categorical_summary[column] = df[column].value_counts().head(10) 

for column, freq in categorical_summary.items():
        print(f"\n{column}的敘述統計總覽:")
        print(freq)


# In[5]:


df["是否死亡"] = ((df["24小時內死亡人數"].fillna(0) + df["30日內死亡人數"].fillna(0)) > 0).astype(int)

df_death = df[df["是否死亡"] == 1].copy()
df_death["是否為A1事故"] = (df_death["24小時內死亡人數"].fillna(0) > 0).astype(int)

# 確認當場 vs. 非當場死亡的樣本數
print(df_death.shape)
df_death


# In[6]:


print(df_death["是否為A1事故"].value_counts())


# In[7]:


categorical_cols = df_death.select_dtypes(include=['object']).columns
categorical_summary = {}

excluded_columns = ['總編號', '案件類別']
categorical_cols = [col for col in categorical_cols if col not in excluded_columns]

for column in categorical_cols:
    value_counts = df_death[column].value_counts()
    total = value_counts.sum()
    # 建立含百分比的顯示格式
    formatted = value_counts.apply(lambda x: f"{x} ({x / total:.1%})")
    categorical_summary[column] = formatted

# 印出結果
for column, freq in categorical_summary.items():
    print(f"\n{column}的敘述統計總覽:")
    print(freq)


# In[33]:


print(df_death["是否為A1事故"].value_counts())
import matplotlib.pyplot as plt

# 計算各類別的百分比
value_counts = df_death["是否為A1事故"].value_counts(normalize=True) * 100

# 畫水平長條圖
plt.figure(figsize=(6, 3))
bars = plt.barh(value_counts.index.astype(str), value_counts.values, color='skyblue')

# 加上百分比標籤
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

plt.title("是否為A1事故 - 百分比分布")
plt.xlabel("百分比 (%)")
plt.ylabel("是否為A1事故")
plt.xlim(0, max(value_counts.values) + 10)  # 預留空間給文字
plt.tight_layout()
plt.show()


# In[34]:


import matplotlib.pyplot as plt

# categorical_cols 已是 list，直接加
all_columns = categorical_cols + ["是否為A1事故"]

fig, axes = plt.subplots(nrows=len(all_columns), ncols=1, figsize=(16, len(all_columns) * 6))

for i, column in enumerate(all_columns):
    ax = axes[i]
    value_counts = df_death[column].value_counts(normalize=True) * 100  # 顯示完整類別百分比

    bars = ax.barh(value_counts.index.astype(str), value_counts.values, color='skyblue')
    ax.set_title(f"{column} 類別統計（完整類別）")
    ax.set_xlabel("百分比 (%)")
    ax.invert_yaxis()

    # 加上百分比標籤
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va='center')

plt.tight_layout()
plt.show()


# In[46]:


import matplotlib.pyplot as plt

# 要繪製的欄位（類別欄位 + 目標變數）
all_columns = categorical_cols + ["是否為A1事故"]

for column in all_columns:
    value_counts = df_death[column].value_counts(normalize=True).head(5) * 100  # 取前五大類別

    # 建立固定大小圖表
    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.barh(value_counts.index.astype(str), value_counts.values, color='skyblue')
    ax.set_title(f"{column} 前五大類別統計")
    ax.set_xlabel("百分比 (%)")
    ax.invert_yaxis()

    # 加上百分比標籤
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va='center')

    plt.tight_layout()
    plt.show()


# In[8]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[9]:


features = ['天候', '道路照明設備', '道路型態', '事故位置', '號誌-號誌種類', '事故類型及型態',"初步分析研判子類別-主要"]
target = '是否為A1事故'

X = df_death[features]
y = df_death[target]
# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot 編碼（不丟棄任何 dummy）
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  
X_train_transformed = encoder.fit_transform(X_train)
X_test_transformed = encoder.transform(X_test)


# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
plt.rcParams['font.family'] = 'Microsoft YaHei'
# 計算 Cramér's V
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

# 計算特徵之間的 Cramér's V
cramers_v_matrix = pd.DataFrame(np.zeros((X.shape[1], X.shape[1])), columns=X.columns, index=X.columns)

for col1 in X.columns:
    for col2 in X.columns:
        cramers_v_matrix.loc[col1, col2] = cramers_v(X[col1], X[col2])

# 畫出 Cramér's V 的熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=0, vmax=1)
plt.title('Cramér\'s V Heatmap of Categorical Features')
plt.show()


# In[11]:


features = ['天候', '道路照明設備', '事故位置', '號誌-號誌種類', '事故類型及型態']
target = '是否為A1事故'

X = df_death[features]
y = df_death[target]

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot 編碼（不丟棄任何 dummy）
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  
X_train_transformed = encoder.fit_transform(X_train)
X_test_transformed = encoder.transform(X_test)

# 計算特徵之間的 Cramér's V
cramers_v_matrix = pd.DataFrame(np.zeros((X.shape[1], X.shape[1])), columns=X.columns, index=X.columns)

for col1 in X.columns:
    for col2 in X.columns:
        cramers_v_matrix.loc[col1, col2] = cramers_v(X[col1], X[col2])

# 畫出 Cramér's V 的熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=0, vmax=1)
plt.title('Cramér\'s V Heatmap of Categorical Features')
plt.show()


# In[12]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_test_transformed)

print("準確率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[13]:


feature_names = encoder.get_feature_names_out(features)
coefficients = model.coef_[0]

# 建立係數的 Series，便於分析
feature_importance = pd.Series(coefficients, index=feature_names).sort_values(ascending=False)

# 顯示前後五個影響因子
print("影響最大的正向因子：")
print(feature_importance.head())

print("\n影響最大的負向因子：")
print(feature_importance.tail())


# In[14]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 設定字體（適用於顯示中文）
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 獲取正確對應的特徵名稱與係數
feature_names = encoder.get_feature_names_out(features)
coefficients = model.coef_[0]

# 建立排序的索引
sorted_indices = np.argsort(coefficients)
sorted_features = feature_names[sorted_indices]
sorted_coefficients = coefficients[sorted_indices]

# 繪製條形圖
plt.figure(figsize=(14, 16))
plt.barh(sorted_features, sorted_coefficients, color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Coefficients')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[15]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 預測機率
y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

# 計算 ROC 曲線
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 繪圖
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


# In[16]:


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# 計算 Precision-Recall 曲線數據
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
ap_score = average_precision_score(y_test, y_pred_proba)

print(f"Average Precision (AUPRC): {ap_score:.4f}")
# 繪製 Precision-Recall 曲線
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AUPRC = {ap_score:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


# In[17]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# 計算混淆矩陣
cm = confusion_matrix(y_test, model.predict(X_test_transformed))

# 繪製混淆矩陣
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: 0', 'Pred: 1'], yticklabels=['True: 0', 'True: 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[18]:


# 取得係數
coefficients = model.coef_[0]

# 取得截距
intercept = model.intercept_[0]

# 印出回歸方程式
equation = f"Logistic Regression: log(p/(1-p)) = {intercept:.4f} + " + " + ".join([f"{coeff:.4f}*X{i}" for i, coeff in enumerate(coefficients)])
print(equation)


terms = [f"{coef:+.4f}*{name}" for coef, name in zip(coefficients, feature_names)]

formula = f"log(p / (1 - p)) = {intercept:+.4f} " + " ".join(terms)
print(formula)
# 只印出係數的數字
formatted_coeffs = [f"{coef:.4f}" for coef in coefficients]
print(" ".join(formatted_coeffs))


# In[19]:


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


# In[20]:


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

        # 計算 AIC 或 BIC
        if criterion == 'aic':
            criterion_value = compute_aic(model, X_train[features], y_train)
        elif criterion == 'bic':
            criterion_value = compute_bic(model, X_train[features], y_train)
        else:
            raise ValueError("criterion 必須是 'aic' 或 'bic'")

        # 評估指標（在 test set 上）
        y_pred = model.predict(X_test[features])
        if metric == 'f1':
            score_metric = f1_score(y_test, y_pred, zero_division=0)
        elif metric == 'balanced_accuracy':
            score_metric = balanced_accuracy_score(y_test, y_pred)
        else:
            raise ValueError("metric 必須是 'f1' 或 'balanced_accuracy'")

        # 綜合評分
        normalized_crit = criterion_value / 1000
        score = weight_criterion * normalized_crit - weight_metric * score_metric

        if score < best_score:
            best_score = score
            best_model = model
            best_features = features.copy()

        # 移除影響最小的特徵
        coef_abs = np.abs(model.coef_[0])
        min_coef_idx = np.argmin(coef_abs)
        features.pop(min_coef_idx)

    return best_model, best_features, best_score

criterion = 'aic'  # 你呼叫時用的criterion

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

# 評估

y_pred_final = final_model.predict(X_test_df[selected])
print("準確率:", accuracy_score(y_test, y_pred_final))
print("\n分類報告：")
print(classification_report(y_test, y_pred_final))


# In[21]:


# 取得係數
coefficients = final_model.coef_[0]

# 取得截距
intercept = final_model.intercept_[0]

# 印出回歸方程式
equation = f"Logistic Regression: log(p/(1-p)) = {intercept:.4f} + " + " + ".join([f"{coeff:.4f}*X{i}" for i, coeff in enumerate(coefficients)])
print(equation)


terms = [f"{coef:+.4f}*{name}" for coef, name in zip(coefficients, X_test_df[selected])]

formula = f"log(p / (1 - p)) = {intercept:+.4f} " + " ".join(terms)
print(formula)

formatted_coeffs1 = [f"{coef:.4f}" for coef in coefficients]
print(" ".join(formatted_coeffs1))


# In[22]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_pred_proba2 = final_model.predict_proba(X_test_df[selected])[:, 1]

# 計算 ROC 曲線資料
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba2)

# AUC 分數
auc_score = roc_auc_score(y_test, y_pred_proba2)

# 繪圖
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # 隨機分類器基準線
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[23]:


# 計算 Precision-Recall 曲線數據
precision2, recall2, _ = precision_recall_curve(y_test, y_pred_proba2)
ap_score2 = average_precision_score(y_test, y_pred_proba2)

print(f"Average Precision (AUPRC): {ap_score2:.4f}")
# 繪製 Precision-Recall 曲線
plt.figure(figsize=(8, 6))
plt.plot(recall2, precision2, color='darkorange', lw=2, label=f'Precision-Recall curve (AUPRC = {ap_score2:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


# In[24]:


import seaborn as sns

coef = final_model.coef_[0]
importance = pd.Series(coef, index=selected)

# 畫出重要性
plt.figure(figsize=(10, 8))
importance.sort_values(ascending=True).plot(kind='barh')
plt.title("Feature Importance (Coefficient Magnitude)")
plt.xlabel("Coefficient")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[25]:


importance


# In[26]:


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# 計算第一條曲線
precision1, recall1, _ = precision_recall_curve(y_test, y_pred_proba)
ap_score1 = average_precision_score(y_test, y_pred_proba)

# 計算第二條曲線
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


# In[27]:


# ROC 曲線資料與 AUC 分數
fpr1, tpr1, _ = roc_curve(y_test, y_pred_proba)
fpr2, tpr2, _ = roc_curve(y_test, y_pred_proba2)
auc1 = roc_auc_score(y_test, y_pred_proba)
auc2 = roc_auc_score(y_test, y_pred_proba2)

# 畫圖
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


# In[28]:


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

results = []

# 設定搜尋範圍
criteria = ['aic', 'bic']
metrics = ['f1', 'balanced_accuracy']
weights = [round(w, 2) for w in np.linspace(0, 1, 21)]  # 0.00 ~ 1.00 間隔 0.05

# 逐組嘗試所有參數組合
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

                # 計算評估指標
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

# 轉為 DataFrame
results_df = pd.DataFrame(results)


# In[29]:


results_df


# In[30]:


best_aic = results_df.loc[results_df['aic'].idxmin()]
best_bic = results_df.loc[results_df['bic'].idxmin()]
best_acc = results_df.loc[results_df['accuracy'].idxmax()]
best_f1 = results_df.loc[results_df['f1'].idxmax()]
best_bacc = results_df.loc[results_df['balanced_accuracy'].idxmax()]
best_roc = results_df.loc[results_df['roc_auc'].idxmax()]
best_auprc = results_df.loc[results_df['auprc'].idxmax()]
best_composite = results_df.loc[results_df['composite_score'].idxmin()]  # 最小的總分

# 彙整成表格查看
summary_df = pd.DataFrame([
    best_aic, best_bic, best_acc, best_f1, best_bacc, best_roc, best_auprc, best_composite
], index=[
    'AIC 最小', 'BIC 最小', '準確率最高', 'F1 分數最高', '平衡準確率最高', 'ROC AUC 最高', 'AUPRC 最高', '綜合評分最佳'
])

# 顯示精簡欄位
summary_df = summary_df[['criterion', 'metric', 'weight_c', 'weight_m',
                         'accuracy', 'f1', 'balanced_accuracy', 'roc_auc', 'auprc',
                         'aic', 'bic', 'num_features']]

summary_df


# In[32]:


results_df.to_excel("results.xlsx", index=False)


# In[ ]:




