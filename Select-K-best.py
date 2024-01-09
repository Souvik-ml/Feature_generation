from sklearn.feature_selection import SelectKBest, f_regression
len(x.columns)
xgbr = XGBRegressor()
gbr = GradientBoostingRegressor()
RFR = RandomForestRegressor()
DTR = DecisionTreeRegressor()
import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score
model = RFR
num_repetitions = 5
num_folds = 10
mae = []
r2 = []
Num_features = []
for i in range(5,106,5):
  Num_features.append(i)
  k_features = i  # Number of top features to select using SIS
  selector = SelectKBest(score_func=f_regression, k=k_features)
  x_selected = selector.fit_transform(x, y)
  rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repetitions, random_state=None)

# Perform cross-validation for MAE and R2 scores
  mae_scores = cross_val_score(model, x_selected, y, scoring='neg_mean_absolute_error', cv=rkf, n_jobs=None)
  mae_scores = -mae_scores
  mae.append(np.mean(mae_scores))
  r2_scores = cross_val_score(model, x_selected, y, scoring='r2', cv=rkf, n_jobs=None)
  r2.append(np.mean(r2_scores))

plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
#plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
plt.rcParams["figure.autolayout"] = True
#plt.rcParams['font.size'] = 11.5

# Plot the data
#plt.plot(Num_features, mae, marker='o', linestyle='-')
#plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
# Define a list of colors for the markers
colors = colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'gray', 'brown',
          'lime', 'pink', 'cyan', 'gold', 'darkgreen', 'skyblue', 'indigo', 'crimson', 'lightcoral', 'teal']
line_color = '#000080'
# Plot the data with different colors for each marker
for i in range(len(Num_features)):
    plt.plot(Num_features[i], mae[i], marker='o', markersize=15, linestyle='-', color=colors[i])
    if i > 0:
        plt.plot([Num_features[i - 1], Num_features[i]], [mae[i - 1], mae[i]], color=line_color)

# Customize the plot
plt.xlabel('Number of Features', fontsize=20)
plt.ylabel('Mean Absolute Error (eV)', fontsize=20)
#plt.title('Number of Features vs. Mean Absolute Error')
plt.grid(True)
plt.xticks(range(0, max(Num_features) + 1, 10))
# Add any additional customizations as needed, such as axis limits, legend, etc.

# Show the plot
plt.tight_layout()  # Adjusts the spacing for better layout
plt.savefig('RFR_Error_num_features2.png', dpi=700, bbox_inches="tight", transparent=True)
plt.show()

features_name = list(x.columns)
print(features_name)

selected_feature_indices = selector.get_support()

selected_feature_names = [features_name[i] for i, selected in enumerate(selected_feature_indices) if selected]

print("Selected Feature Names:", selected_feature_names)