# Exploring-Key-Factors-and-Constructing-a-Predictive-Model-for-Traffic-Accidents-in-Tainan-City
Final report of Big Data Analytics and Data Mining, June 2025@ NCKU Resource Engineering

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)
![Colab](https://img.shields.io/badge/Google-Colab-yellow?logo=googlecolab)


- Explored key predictors of fatal traffic accidents in Tainan City using logistic regression.  
- Applied Cramér’s V for categorical variable screening and evaluated model performance using AIC, BIC, accuracy, and F1 score.  
- Successfully identified a simplified, interpretable predictive model with key significant variables using Python.

## Project Objectives

- Explore key predictors of A1 fatal traffic accidents in Tainan City
- Apply Cramér’s V for categorical variable association screening
- Develop both full logistic regression and simplified interpretable models
- Evaluate models using AIC, BIC, Accuracy, F1, ROC AUC, AUPRC
- Identify actionable road-safety insights for practical use

## Dataset Description
<div align="center">

  | Item	| Value |
|-----------|-------|
|Source	|Tainan City Open Data Platform|
|Year	|2024 |
|Total Records (Merged)	|44,177|
|Fatal Accidents Extracted	|209|
|A1 Accidents (death in 24 hours)	|150|
|A2 Accidents	|59|

</div>
Only fatal accidents (A1 + A2) were used for modeling.

## Methodology Overview
### 1. Data Merging & Cleaning

- Combined two releases of Tainan accident data (1130106 + 1130712)
- Selected accidents with:
<div align="center">
<code>24小時內死亡人數 + 30日內死亡人數 > 0</code>
</div>

- Defined target variable:
<div align="center">
  <code>A1 = 1 if 24小時內死亡人數 > 0, else 0</code>
</div>
### 2. Categorical Variable Analysis

- Applied value counts, distribution plots
- Calculated Cramér’s V correlation matrix
- Removed redundant categorical fields (> 0.5 correlation)
<img width="1097" height="976" alt="圖片" src="https://github.com/user-attachments/assets/a2909fdf-db0c-41e2-94a8-1291018c9eca" />

### 3. Feature Engineering

- One-hot encoded all usable categorical fields
- Final full logistic model: 39 features

### 4. Model Construction

Logistic Regression with:
- Balanced class weights
- 5-fold internal evaluation

Developed:

- Full Model (39 vars)
- Simplified Model (10 vars) using backward elimination

$$Objective = Weighted combination of AIC / BIC / Accuracy / F1$$

### 5. Model Evaluation

- Accuracy
Precision, Recall, F1
- ROC AUC
AUPRC (important due to class imbalance)
- Information criteria: AIC, BIC

## Results
### Performance Comparison

Full Model (39 features)

<div align="center">
  
|Metric|Value|
|-----------|-------------|
|Accuracy   | 0.6904|
|ROC AUC   | 0.6811|
|AUPRC      | 0.8607|
|AIC       | 231.71|
|BIC        | 356.43|
</div>

<img width="1194" height="463" alt="圖片" src="https://github.com/user-attachments/assets/ca755369-8209-4997-bf95-a647841b6316" />


Simplified Model (10 features)
<div align="center">
  
|Metric|Value|
|------------|-----------------|
|Accuracy   | 0.7378|
|ROC AUC    | 0.6856|
|AUPRC      | 0.8520|
|AIC        | 180.77|
|BIC        | 215.06|
</div>

<img width="1196" height="430" alt="圖片" src="https://github.com/user-attachments/assets/9e570e3a-0c0f-41c0-b469-95d4bc62d476" />


### Selected Top 10 Predictive Features
1. 天候：陰 (Weather_Cloudy)
2. 道路照明：有照明但未開啟或故障 (Lighting_Faulty)
3. 道路照明：無照明 (Lighting_None)
4. 行人穿越道 (Location_Crosswalk)
5. 路肩／路緣 (Location_Shoulder)
6. 閃光號誌 (Signal_Flashing)
7. 型態：其他 (Type_Other)
8. 撞護欄（樁）(Type_Guardrail)
9. 撞電桿 (Type_Pole)
10. 衝出路外 (Type_RoadDeparture)

Simplified Logistic Regression Equation
$$logit(p) =0.6296+ 0.6507 * (WeatherCloudy)+ 0.8756 * (LightingFaulty)- 0.8444 * (LightingNone)+ 0.8310 * (LocationCrosswalk)+ 0.6159 * (Location_Shoulder)+ 0.9158 * (SignalFlashing)- 1.5743 * (TypeOther)+ 0.7974 * (TypeGuardrail)+ 0.8970 * (TypePole)- 0.7148 * (TypeRoadDeparture)$$

Confusion Matrix (Simplified Model)
| Actual\Predicted | 0  | 1  |
|-----------------|----|----|
| 0               | 34 | 15 |
| 1               | 18 | 83 |

Interpretation Highlights
- Nighttime risk: Faulty or inactive road lighting is one of the strongest predictors of A1 fatal accidents.
- Flashing signals: Intersections controlled by flashing signals have a higher likelihood of resulting in A1 fatal crashes compared to standard traffic signals.
- Pedestrian risk: Accidents occurring at pedestrian crossings are associated with a higher probability of fatal outcomes.
- Roadside hazards: Roadside structures such as guardrails and utility poles show strong associations with A1 fatal accidents.
- Model simplification: After feature reduction, the AIC/BIC values dropped significantly while maintaining performance, resulting in a more compact and interpretable model.



### Key Insights for Traffic Policy

- **Improve nighttime lighting**

  Faulty or inactive lighting is strongly linked to fatal outcomes.

- **Replace or supplement flashing signals**

  Flashing-only intersections show significantly higher risk.

- **Enhance pedestrian crossing safety**

  Consider raised crosswalks, refuge islands, and illuminated markings.

- **Mitigate roadside hazards**

  Relocate or shield electric poles and guardrails where feasible.

-  **Use simplified model for city-level monitoring**

   Only 10 variables needed—lightweight and interpretable.

### Conclusion

This project successfully builds an interpretable logistic regression model that identifies key risk factors associated with fatal traffic accidents in Tainan City. Through categorical screening, feature engineering, and backward elimination, we obtain a simplified yet highly effective model with strong predictive performance and actionable policy implications.

### Reference
Akaike, H. （1974）. A new look at the statistical model identification. IEEE Transactions on Automatic Control, 19（6）, 716–723. https://doi.org/10.1109/TAC.1974.1100705

Agresti, A. （2019）. An introduction to categorical data analysis （3rd ed.）.Hoboken, NJ: John Wiley & Sons.

Burnham, K. P., & Anderson, D. R. （2002）. Model selection and multimodel inference: A practical information-theoretic approach （2nd ed.）. Springer.

Chen, C.-M. (2023, February 6). Highest in eight years! Nearly 3,000 traffic deaths in 2022: This county is the real pedestrian hell. Yahoo News. https://ynews.page.link/91AtA

Cramér, H. （1946）. Mathematical methods of statistics. Princeton University Press.

Hastie, T., Tibshirani, R., & Friedman, J. （2009）. The elements of statistical learning: Data mining, inference, and prediction （2nd ed.）. Springer. https://doi.org/10.1007/978-0-387-84858-7

Hu, J.-L. (2025, March 5). 2024 road safety report: Kaohsiung records 310 traffic deaths, the highest in Taiwan. United Daily News. https://udn.com/news/story/7266/8587926

Pan, Y. (2023, November 28). 2023's most dangerous city for traffic accidents is not Tainan! Latest statistics revealed. NOWnews. https://www.nownews.com/news/6310617

Schwarz, G. （1978）. Estimating the dimension of a model. The Annals of Statistics, 6（2）, 461– 464. https://doi.org/10.1214/aos/1176344136

Tainan City Police Department. （2024）. Tainan traffic accident causes and casualties statistics dataset [Data set]. Tainan City Open Data Platform. https://data.tainan.gov.tw/dataset/policedata016
