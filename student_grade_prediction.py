import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import pickle
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
    SGDClassifier,
    BayesianRidge,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from six import StringIO
from IPython.display import Image
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import LinearSVC
from numpy.ma.core import sqrt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from numpy.polynomial.polynomial import polyfit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    confusion_matrix
)

warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('/content/student-mat.csv')

df.shape

df.head()

df.tail()

df.info()

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(categorical_columns)
print(len(categorical_columns))

numerical_columns = df.select_dtypes(include=['int64']).columns.tolist()
print(numerical_columns)
print(len(numerical_columns))

df.describe().T

df.describe(include='object')

plt.figure(figsize=(6, 4))
sns.set_theme(style="ticks")
sns.pairplot(df, hue="G3")

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot for G1
b1 = sns.countplot(x='G1', data=df, ax=axs[0])
b1.set_title('The distribution of grade 1 of students', fontsize=16)
b1.set_xlabel('Grade 1', fontsize=14)
b1.set_ylabel('Count', fontsize=14)
# Plot for G2
b2 = sns.countplot(x='G2', data=df, ax=axs[1])
b2.set_title('The distribution of grade 2 of students', fontsize=16)
b2.set_xlabel('Grade 2', fontsize=14)
b2.set_ylabel('Count', fontsize=14)

# Plot for G3
b3 = sns.countplot(x='G3', data=df, ax=axs[2])
b3.set_title('The distribution of final grade of students', fontsize=16)
b3.set_xlabel('Final Grade', fontsize=14)
b3.set_ylabel('Count', fontsize=14)
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

plt.figure(figsize=(12,4))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset');

import matplotlib.pyplot as plt

# Calculate percentage values
total = len(df)
value_counts = df['school'].value_counts()
percentages = value_counts / total * 100

# Rename the labels from 'GP' and 'MS' to custom names
value_counts.index = ['Gabriel Pereira School', 'Mousinho da Silveira School']
# Define custom colors
custom_colors = ['#ff5733', '#66b3ff']  # A vibrant red and a bright blue

# Define an explode parameter to separate one of the sections
explode = (0.1, 0)  # This will explode the first section (Gabriel Pereira School) slightly

# Create a pie chart with custom colors and explode parameter
plt.figure(figsize=(5, 5))
plt.pie(
    percentages,
    labels=value_counts.index,
    autopct=lambda p: f'{p:.1f}%',
    startangle=140,
    colors=custom_colors,
    explode=explode  # Add the explode parameter here
)
# Equal aspect ratio ensures that the pie is drawn as a circle
plt.axis('equal')

# Add a title
plt.title("Distribution of Students' Schools")

# Show the plot
plt.tight_layout()
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Grade distribution by school for G1
sns.kdeplot(df.loc[df['school'] == 'GP', 'G1'], label='GP', shade=True, ax=axs[0])
sns.kdeplot(df.loc[df['school'] == 'MS', 'G1'], label='MS', shade=True, ax=axs[0])
axs[0].set_title('GP vs. MS - First Period Grade', fontsize=16)
axs[0].set_xlabel('Grade 1', fontsize=14)
axs[0].set_ylabel('Density', fontsize=14)

# Grade distribution by school for G2
sns.kdeplot(df.loc[df['school'] == 'GP', 'G2'], label='GP', shade=True, ax=axs[1])
sns.kdeplot(df.loc[df['school'] == 'MS', 'G2'], label='MS', shade=True, ax=axs[1])
axs[1].set_title('GP vs. MS - Second Period Grade', fontsize=16)
axs[1].set_xlabel('Grade 2', fontsize=14)
axs[1].set_ylabel('Density', fontsize=14)
# Grade distribution by school for G3
sns.kdeplot(df.loc[df['school'] == 'GP', 'G3'], label='GP', shade=True, ax=axs[2])
sns.kdeplot(df.loc[df['school'] == 'MS', 'G3'], label='MS', shade=True, ax=axs[2])
axs[2].set_title('GP vs. MS - Final Grade', fontsize=16)
axs[2].set_xlabel('Final Grade', fontsize=14)
axs[2].set_ylabel('Density', fontsize=14)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Calculate percentage values
total = len(df)
value_counts = df['sex'].value_counts()
percentages = value_counts / total * 100

# Rename the labels from 'F' and 'M' to custom names
value_counts.index = ['Female', 'Male']

# Define custom colors
custom_colors = ['#ff9999', '#66b3ff']  # You can change these colors as desired
# Define an explode parameter to separate one of the sections
#explode = (0.061, 0)  # This will explode the first section (Female) slightly

# Create a pie chart with custom colors and explode parameter
plt.figure(figsize=(4, 4))
plt.pie(
    percentages,
    labels=value_counts.index,
    autopct=lambda p: f'{p:.1f}%',
    startangle=140,
    colors=custom_colors,
    #explode=explode  # Add the explode parameter here
)
# Equal aspect ratio ensures that the pie is drawn as a circle.
plt.axis('equal')

# Add a title
plt.title("Ratio of Students' gender at Schools", fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

# Define custom colors
custom_colors = ['#ff9999', '#004c99']  # Darker blue color for the second bar, you can change these colors as desired

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot for G1 with custom colors
sns.barplot(x='sex', y='G1', data=df, ci=None, ax=axs[0], palette=custom_colors)
axs[0].set_xlabel('Gender', fontsize=14)
axs[0].set_ylabel('Grade 1', fontsize=14)
axs[0].set_xticks([0, 1])
axs[0].set_xticklabels(["Female", "Male"])
# Plot for G2 with custom colors
sns.barplot(x='sex', y='G2', data=df, ci=None, ax=axs[1], palette=custom_colors)
axs[1].set_xlabel('Gender', fontsize=14)
axs[1].set_ylabel('Grade 2', fontsize=14)
axs[1].set_xticks([0, 1])
axs[1].set_xticklabels(["Female", "Male"])

# Plot for G3 with custom colors
sns.barplot(x='sex', y='G3', data=df, ci=None, ax=axs[2], palette=custom_colors)
axs[2].set_xlabel('Gender', fontsize=14)
axs[2].set_ylabel('Final Grade', fontsize=14)
axs[2].set_xticks([0, 1])
axs[2].set_xticklabels(["Female", "Male"])
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

custom_colors = ['#ff9999', '#66b3ff']  # You can change these colors as desired
sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=df, kind="bar",
    x="sex", y="G3", hue="school",
    errorbar="sd", height=6, ci=None
)
g.despine(left=True)
custom_xtitles = ["Female", "Male"]
g.set_xticklabels(custom_xtitles)
g.set_axis_labels("", "Final Grade")
g.legend.set_title("")

# Set a custom color palette
custom_palette = sns.color_palette("colorblind")

# Set the Seaborn style with the custom color palette
sns.set_theme(style="whitegrid", palette=custom_palette)

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="school", y="G3", hue="sex",
    errorbar="sd", height=6, ci=None
)
g.despine(left=True)
new_x_labels = ["Gabriel Pereira School", "Mousinho da Silveira School"]  # Modify these labels as needed
plt.xticks(range(len(new_x_labels)), new_x_labels)

g.set_axis_labels("", "Final Grade")

g.legend.set_title("")

b = sns.countplot(x='age', hue='sex', data=df)
b.axes.set_title('Number of students in different age groups', fontsize=16)
b.set_xlabel("Age", fontsize=14)
b.set_ylabel("Count", fontsize=14)
plt.show()

b = sns.swarmplot(x='age', y='G3',hue='sex', data=df)
b.set_xlabel('Age', fontsize = 14)
b.set_ylabel('Final Grade', fontsize = 14)
plt.show()

import matplotlib.pyplot as plt

# Calculate percentage values
total = len(df)
value_counts = df['address'].value_counts()
percentages = value_counts / total * 100

# Define custom colors for each part
colors = ['#3399ff', '#ff6600']

# Define the amount of space between the parts
explode = (0.1, 0)

# Define custom labels for the parts
custom_labels = ['Urban', 'Rural']
# Create a pie chart
plt.figure(figsize=(4, 4))  # Adjust the figure size if needed
plt.pie(
    percentages,
    labels=custom_labels,
    autopct=lambda p: f'{p:.1f}%',
    startangle=140,
    colors=colors,  # Use custom colors
    explode=explode,  # Add space between parts
)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')
# Update the title
plt.title("Ratio of Students' Addresses to Urban and Rural Areas", fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Grade 1 plot
sns.barplot(x=df['address'], y=df['G1'], data=df, ci=None, ax=axs[0])
axs[0].set_xlabel('Address', fontsize=14)
axs[0].set_ylabel('Grade 1', fontsize=14)
axs[0].set_xticklabels(["Urban", "Rural"])

# Grade 2 plot
sns.barplot(x=df['address'], y=df['G2'], data=df, ci=None, ax=axs[1])
axs[1].set_xlabel('Address', fontsize=14)
axs[1].set_ylabel('Grade 2', fontsize=14)
axs[1].set_xticklabels(["Urban", "Rural"])
# Final Grade plot
sns.barplot(x=df['address'], y=df['G3'], data=df, ci=None, ax=axs[2])
axs[2].set_xlabel('Address', fontsize=14)
axs[2].set_ylabel('Final Grade', fontsize=14)
axs[2].set_xticklabels(["Urban", "Rural"])

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

custom_palette = ["#FF5733", "#33FF57", "#3357FF", "#FF33B5"]
b = sns.swarmplot(x='reason', y='G3', data=df,palette=custom_palette)
b.set_xlabel('Reason', fontsize = 14)
b.set_ylabel('Final Grade', fontsize = 14)
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Grade 1 plot
b1 = sns.barplot(x=df['famsize'], y=df['G1'], data=df, ci=None, ax=axs[0])
b1.set_xlabel('Family Size', fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)
b1.set_xticklabels(["Greater Than Three", "Less Than Three"])

# Grade 2 plot
b2 = sns.barplot(x=df['famsize'], y=df['G2'], data=df, ci=None, ax=axs[1])
b2.set_xlabel('Family Size', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(["Greater Than Three", "Less Than Three"])
# Final Grade plot
b3 = sns.barplot(x=df['famsize'], y=df['G3'], data=df, ci=None, ax=axs[2])
b3.set_xlabel('Family Size', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(["Greater Than Three", "Less Than Three"])

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Grade 1 plot
b1 = sns.barplot(x=df['Pstatus'], y=df['G1'], data=df, ci=None, ax=axs[0])
b1.set_xlabel("Parent's Cohabitation Status", fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)
b1.set_xticklabels(["Apart", "Living Together"])

# Grade 2 plot
b2 = sns.barplot(x=df['Pstatus'], y=df['G2'], data=df, ci=None, ax=axs[1])
b2.set_xlabel("Parent's Cohabitation Status", fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(["Apart", "Living Together"])
# Final Grade plot
b3 = sns.barplot(x=df['Pstatus'], y=df['G3'], data=df, ci=None, ax=axs[2])
b3.set_xlabel("Parent's Cohabitation Status", fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(["Apart", "Living Together"])

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
education_descriptions = [
    "0 - None",
    "1 - Primary (4th Grade)",
    "2 - 5th to 9th Grade",
    "3 - Secondary Education",
    "4 - Higher Education"
]
# Grade 1 plot
b1 = sns.barplot(x='Fedu', y='G1', data=df, ci=None, ax=axs[0])
b1.set_xlabel("Father Education Level", fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)
b1.text(0.05, -0.35, '\n'.join(education_descriptions), fontsize=12, transform=plt.gcf().transFigure)
# Grade 2 plot
b2 = sns.barplot(x='Fedu', y='G2', data=df, ci=None, ax=axs[1])
b2.set_xlabel("Father Education Level", fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.text(0.05, -0.35, '\n'.join(education_descriptions), fontsize=12, transform=plt.gcf().transFigure)

# Final Grade plot
b3 = sns.barplot(x='Fedu', y='G3', data=df, ci=None, ax=axs[2])
b3.set_xlabel("Father Education Level", fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.text(0.05, -0.35, '\n'.join(education_descriptions), fontsize=12, transform=plt.gcf().transFigure)
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Grade 1 plot
b1 = sns.barplot(x=df['Medu'], y=df['G1'], data=df, ci=None, ax=axs[0])
b1.set_xlabel("Mother's Education Level", fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)

# Grade 2 plot
b2 = sns.barplot(x=df['Medu'], y=df['G2'], data=df, ci=None, ax=axs[1])
b2.set_xlabel("Mother's Education Level", fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
# Final Grade plot
b3 = sns.barplot(x=df['Medu'], y=df['G3'], data=df, ci=None, ax=axs[2])
b3.set_xlabel("Mother's Education Level", fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)

# Adjust layout
plt.tight_layout()
# Show the plots
plt.show()

# Create a figure with three subplots arranged in a row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot the first chart
b1 = sns.barplot(x=df['Fjob'], y=df['G1'], data=df, ax=ax1, errcolor='0.2', capsize=0.2)
b1.set_xlabel('Father Job Title', fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)
b1.set_xticklabels(["Teacher", "Other", "Services", "Health", "At Home"])

# Plot the second chart
b2 = sns.barplot(x=df['Fjob'], y=df['G2'], data=df, ax=ax2, errcolor='0.2', capsize=0.2)
b2.set_xlabel('Father Job Title', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(["Teacher", "Other", "Services", "Health", "At Home"])
# Plot the third chart
b3 = sns.barplot(x=df['Fjob'], y=df['G3'], data=df, ax=ax3, errcolor='0.2', capsize=0.2)
b3.set_xlabel('Father Job Title', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(["Teacher", "Other", "Services", "Health", "At Home"])

# Adjust the layout
plt.tight_layout()

# Show the plots
plt.show()

# Create a figure with three subplots arranged in a row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot the first chart
b1 = sns.barplot(x=df['Mjob'], y=df['G1'], data=df, ax=ax1, errcolor=None, capsize=0.2)
b1.set_xlabel('Mother Job Title', fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)
b1.set_xticklabels(["At Home", "Health", "Other", "Services", "Teacher"])

# Plot the second chart
b2 = sns.barplot(x=df['Mjob'], y=df['G2'], data=df, ax=ax2, errcolor=None, capsize=0.2)
b2.set_xlabel('Mother Job Title', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(["At Home", "Health", "Other", "Services", "Teacher"])
# Plot the third chart
b3 = sns.barplot(x=df['Mjob'], y=df['G3'], data=df, ax=ax3, errcolor=None, capsize=0.2)
b3.set_xlabel('Mother Job Title', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(["At Home", "Health", "Other", "Services", "Teacher"])

# Adjust the layout
plt.tight_layout()

# Show the plots
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
study_times = [
    "0 - None",
    "1 - Primary (4th Grade)",
    "2 - 5th to 9th Grade",
    "3 - Secondary Education",
    "4 - Higher Education"
]


# Create the first horizontal bar plot
b1 = sns.barplot(x=df['G1'], y=df['studytime'], data=df, ci=None, ax=axs[0], orient='h')
b1.set_ylabel('Study Time', fontsize=14)
b1.set_xlabel('Grade 1', fontsize=14)
b1.set_title('Impact Study Time on grade1', fontsize=16)
b1.text(0.05, -0.35, '\n'.join(study_times), fontsize=12, transform=plt.gcf().transFigure)
# Create the second horizontal bar plot
b2 = sns.barplot(x=df['G2'], y=df['studytime'], data=df, ci=None, ax=axs[1], orient='h')
b2.set_ylabel('Study Time', fontsize=14)
b2.set_xlabel('Grade 2', fontsize=14)
b2.set_title('Impact Study Time on grade 2', fontsize=16)
b2.text(0.05, -0.35, '\n'.join(study_times), fontsize=12, transform=plt.gcf().transFigure)


# Create the third horizontal bar plot
b3 = sns.barplot(x=df['G3'], y=df['studytime'], data=df, ci=None, ax=axs[2], orient='h')
b3.set_ylabel('Study Time', fontsize=14)
b3.set_xlabel('Final Grade', fontsize=14)
b3.set_title('Impact Study Time on final grade', fontsize=16)
b3.text(0.05, -0.35, '\n'.join(study_times), fontsize=12, transform=plt.gcf().transFigure)
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="studytime", y="G1", hue="school",
    errorbar="sd", height=6, ci=None
)
g.despine(left=True)
g.set_axis_labels("Study Time", "Grade 1")
g.legend.set_title("")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="studytime", y="G2", hue="school",
    errorbar="sd", height=6, ci=None
)
g.despine(left=True)
g.set_axis_labels("Study Time", "Grade 2")
g.legend.set_title("")
# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="studytime", y="G3", hue="school",
    errorbar="sd", height=6, ci=None
)
g.despine(left=True)
g.set_axis_labels("Study Time", "Final Grade")
g.legend.set_title("")

import matplotlib.pyplot as plt
import seaborn as sns

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Plot the first chart
b = sns.barplot(x=df['guardian'], y=df['G1'], data=df, ci=None, ax=axes[0])
b.set_xlabel('Guardian', fontsize=14)
b.set_ylabel('Grade 1', fontsize=14)
custom_xtitles = ["Mother", "Father", "Other"]  # Replace with your desired labels
b.set_xticklabels(custom_xtitles)
# Plot the second chart
b2 = sns.barplot(x=df['guardian'], y=df['G1'], data=df, ci=None, ax=axes[1])
b2.set_xlabel('Guardian', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(custom_xtitles)

# Plot the third chart
b3 = sns.barplot(x=df['guardian'], y=df['G1'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('Guardian', fontsize=14)
b3.set_ylabel('Grade 1', fontsize=14)
b3.set_xticklabels(custom_xtitles)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Plot the first chart
b = sns.barplot(x=df['schoolsup'], y=df['G1'], data=df, ci=None, ax=axes[0])
b.set_xlabel('extra educational support', fontsize=14)
b.set_ylabel('Grade 1', fontsize=14)
custom_xtitles = ["Yes", "No"]  # Replace with your desired labels
b.set_xticklabels(custom_xtitles)

# Plot the second chart
b2 = sns.barplot(x=df['schoolsup'], y=df['G2'], data=df, ci=None, ax=axes[1])
b2.set_xlabel('extra educational support', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(custom_xtitles)
# Plot the third chart
b3 = sns.barplot(x=df['schoolsup'], y=df['G3'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('extra educational support', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(custom_xtitles)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Plot the first chart
b = sns.barplot(x=df['famsup'], y=df['G1'], data=df, ci=None, ax=axes[0])
b.set_xlabel('Family Educational Support', fontsize=14)
b.set_ylabel('Grade 1', fontsize=14)
custom_xtitles = ["No", "Yes"]  # Replace with your desired labels
b.set_xticklabels(custom_xtitles)

# Plot the second chart
b1 = sns.barplot(x=df['famsup'], y=df['G2'], data=df, ci=None, ax=axes[1])
b1.set_xlabel('Family Educational Support', fontsize=14)
b1.set_ylabel('Grade 2', fontsize=14)
b1.set_xticklabels(custom_xtitles)

# Plot the third chart
b3 = sns.barplot(x=df['famsup'], y=df['G3'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('Family Educational Support', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(custom_xtitles)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()


# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Plot the first chart
b = sns.barplot(x=df['paid'], y=df['G1'], data=df, ci=None, ax=axes[0])
b.set_xlabel('Extra Paid Classes', fontsize=14)
b.set_ylabel('Grade 1', fontsize=14)
custom_xtitles = ["No", "Yes"]  # Replace with your desired labels
b.set_xticklabels(custom_xtitles, fontsize=12)

# Plot the second chart
b1 = sns.barplot(x=df['paid'], y=df['G2'], data=df, ci=None, ax=axes[1])
b1.set_xlabel('Extra Paid Classes', fontsize=14)
b1.set_ylabel('Grade 2', fontsize=14)
b1.set_xticklabels(custom_xtitles, fontsize=12)
# Plot the third chart
b3 = sns.barplot(x=df['paid'], y=df['G3'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('Extra Paid Classes', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(custom_xtitles, fontsize=12)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Plot the first chart
b = sns.barplot(x=df['activities'], y=df['G1'], data=df, ci=None, ax=axes[0])
b.set_xlabel('Activities', fontsize=14)
b.set_ylabel('Grade 1', fontsize=14)
custom_xtitles = ["No", "Yes"]  # Replace with your desired labels
b.set_xticklabels(custom_xtitles, fontsize=12)
# Plot the second chart
b1 = sns.barplot(x=df['activities'], y=df['G2'], data=df, ci=None, ax=axes[1])
b1.set_xlabel('Activities', fontsize=14)
b1.set_ylabel('Grade 2', fontsize=14)
b1.set_xticklabels(custom_xtitles, fontsize=12)

# Plot the third chart
b3 = sns.barplot(x=df['activities'], y=df['G3'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('Activities', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(custom_xtitles, fontsize=12)

# Adjust the spacing between subplots
plt.tight_layout()
# Display the plots
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Plot the first chart
b = sns.barplot(x='nursery', y='G1', data=df, ci=None, ax=axes[0])
b.set_xlabel('Nursery', fontsize=14)
b.set_ylabel('Grade 1', fontsize=14)
custom_xtitles = ["Yes", "No"]  # Replace with your desired labels
b.set_xticklabels(custom_xtitles, fontsize=12)

# Plot the second chart
b1 = sns.barplot(x='nursery', y='G2', data=df, ci=None, ax=axes[1])
b1.set_xlabel('Nursery', fontsize=14)
b1.set_ylabel('Grade 2', fontsize=14)
b1.set_xticklabels(custom_xtitles, fontsize=12)
# Plot the third chart
b2 = sns.barplot(x='nursery', y='G3', data=df, ci=None, ax=axes[2])
b2.set_xlabel('Nursery', fontsize=14)
b2.set_ylabel('Final Grade', fontsize=14)
b2.set_xticklabels(custom_xtitles, fontsize=12)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Labels for 'higher' variable
custom_xtitles = ["Yes", "No"]

# Plot the first chart
b1 = sns.barplot(x=df['higher'], y=df['G1'], data=df, ci=None, ax=axes[0])
b1.set_xlabel('Wants to take higher education', fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)
b1.set_xticklabels(custom_xtitles, fontsize=12)
# Plot the second chart
b2 = sns.barplot(x=df['higher'], y=df['G2'], data=df, ci=None, ax=axes[1])
b2.set_xlabel('Wants to take higher education', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(custom_xtitles, fontsize=12)

# Plot the third chart
b3 = sns.barplot(x=df['higher'], y=df['G3'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('Wants to take higher education', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(custom_xtitles, fontsize=12)
# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Labels for 'internet' variable
custom_xtitles = ["No", "Yes"]

# Plot the first chart
b1 = sns.barplot(x=df['internet'], y=df['G1'], data=df, ci=None, ax=axes[0])
b1.set_xlabel('Internet Access', fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)
b1.set_xticklabels(custom_xtitles, fontsize=12)
# Plot the second chart
b2 = sns.barplot(x=df['internet'], y=df['G2'], data=df, ci=None, ax=axes[1])
b2.set_xlabel('Internet Access', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(custom_xtitles, fontsize=12)

# Plot the third chart
b3 = sns.barplot(x=df['internet'], y=df['G3'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('Internet Access', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(custom_xtitles, fontsize=12)
# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Labels for 'romantic' variable
custom_xtitles = ["No", "Yes"]

# Plot the first chart
b1 = sns.barplot(x=df['romantic'], y=df['G1'], data=df, ci=None, ax=axes[0])
b1.set_xlabel('Romantic Relationship', fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)
b1.set_xticklabels(custom_xtitles, fontsize=12)
# Plot the second chart
b2 = sns.barplot(x=df['romantic'], y=df['G2'], data=df, ci=None, ax=axes[1])
b2.set_xlabel('Romantic Relationship', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
b2.set_xticklabels(custom_xtitles, fontsize=12)

# Plot the third chart
b3 = sns.barplot(x=df['romantic'], y=df['G3'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('Romantic Relationship', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)
b3.set_xticklabels(custom_xtitles, fontsize=12)

# Adjust the spacing between subplots
plt.tight_layout()
# Display the plots
plt.show()

df_encoded = df.copy()

for col in df_encoded.select_dtypes(include=["object"]).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

plt.figure(figsize=(15, 13))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Plot the first chart for 'G1'
b1 = sns.barplot(x=df['failures'], y=df['G1'], data=df, ci=None, ax=axes[0])
b1.set_xlabel('Failures', fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)

# Plot the second chart for 'G2'
b2 = sns.barplot(x=df['failures'], y=df['G2'], data=df, ci=None, ax=axes[1])
b2.set_xlabel('Failures', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
# Plot the third chart for 'G3'
b3 = sns.barplot(x=df['failures'], y=df['G3'], data=df, ci=None, ax=axes[2])
b3.set_xlabel('Failures', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

b= sns.scatterplot(x=df['G1'],y=df['G3'],data=df)
b.set_xlabel('Grade 1', fontsize = 14)
b.set_ylabel('Final Grade', fontsize = 14)
plt.show()

b= sns.scatterplot(x=df['G2'],y=df['G3'],data=df)
b.set_xlabel('Grade 2', fontsize = 14)
b.set_ylabel('Final Grade', fontsize = 14)
plt.show()

# Create a subplot grid with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed

# Plot the first chart for 'G1'
b1 = sns.lineplot(x='absences', y='G1', hue='school', style='school', data=df, ax=axes[0])
b1.set_xlabel('Absences', fontsize=14)
b1.set_ylabel('Grade 1', fontsize=14)

# Plot the second chart for 'G2'
b2 = sns.lineplot(x='absences', y='G2', hue='school', style='school', data=df, ax=axes[1])
b2.set_xlabel('Absences', fontsize=14)
b2.set_ylabel('Grade 2', fontsize=14)
# Plot the third chart for 'G3'
b3 = sns.lineplot(x='absences', y='G3', hue='school', style='school', data=df, ax=axes[2])
b3.set_xlabel('Absences', fontsize=14)
b3.set_ylabel('Final Grade', fontsize=14)

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()

df.drop(df[df['G3'] < 1].index, inplace = True)

df_ohe = pd.get_dummies(df, drop_first=True)

# Calculate the correlation matrix for all columns
correlation_matrix = df_ohe.corr()

# Extract the correlation values for the 'G3' column
correlation_with_G3 = correlation_matrix['G3']

# Create a heatmap of the correlation values
plt.figure(figsize=(5, 13))
sns.heatmap(correlation_with_G3.to_frame(), annot=True, cbar=True)
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
bars = [
    "0 - Very Low",
    "1 - Low",
    "2 - Moderate",
    "3 - High",
    "4 - Very High"
]


# Create the first horizontal bar plot
b1 = sns.barplot(x='G1', y='goout', data=df_ohe, ci=None, ax=axs[0], orient='h')
b1.set_ylabel('Go Out', fontsize=14)
b1.set_xlabel('Grade 1', fontsize=14)
b1.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)
# Create the second horizontal bar plot
b2 = sns.barplot(x=df['G2'], y=df['goout'], data=df, ci=None, ax=axs[1], orient='h')
b2.set_ylabel('Go Out', fontsize=14)
b2.set_xlabel('Grade 2', fontsize=14)
b2.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)


# Create the third horizontal bar plot
b3 = sns.barplot(x='G3', y='goout', data=df_ohe, ci=None, ax=axs[2], orient='h')
b3.set_ylabel('Go Out', fontsize=14)
b3.set_xlabel('Final Grade', fontsize=14)
b3.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
bars = [
    "0 - Very Low",
    "1 - Low",
    "2 - Moderate",
    "3 - High",
    "4 - Very High"
]


# Create the first horizontal bar plot
b1 = sns.barplot(x='G1', y='Walc', data=df_ohe, ci=None, ax=axs[0], orient='h')
b1.set_ylabel('Weekend Alcohol', fontsize=14)
b1.set_xlabel('Grade 1', fontsize=14)
b1.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)
# Create the second horizontal bar plot
b2 = sns.barplot(x='G2', y='Walc', data=df_ohe, ci=None, ax=axs[1], orient='h')
b2.set_ylabel('Weekend Alcohol', fontsize=14)
b2.set_xlabel('Grade 2', fontsize=14)
b2.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)


# Create the third horizontal bar plot
b3 = sns.barplot(x='G3', y='Walc', data=df_ohe, ci=None, ax=axs[2], orient='h')
b3.set_ylabel('Weekend Alcohol', fontsize=14)
b3.set_xlabel('Final Grade', fontsize=14)
b3.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
bars = [
    "0 - Very Low",
    "1 - Low",
    "2 - Moderate",
    "3 - High",
    "4 - Very High"
]


# Create the first horizontal bar plot
b1 = sns.barplot(x='G1', y='Dalc', data=df_ohe, ci=None, ax=axs[0], orient='h')
b1.set_ylabel('Workday Alcohol', fontsize=14)
b1.set_xlabel('Grade 1', fontsize=14)
b1.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)
# Create the second horizontal bar plot
b2 = sns.barplot(x='G2', y='Dalc', data=df_ohe, ci=None, ax=axs[1], orient='h')
b2.set_ylabel('Workday Alcohol', fontsize=14)
b2.set_xlabel('Grade 2', fontsize=14)
b2.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)


# Create the third horizontal bar plot
b3 = sns.barplot(x='G3', y='Dalc', data=df_ohe, ci=None, ax=axs[2], orient='h')
b3.set_ylabel('Workday Alcohol', fontsize=14)
b3.set_xlabel('Final Grade', fontsize=14)
b3.text(0.05, -0.35, '\n'.join(bars), fontsize=12, transform=plt.gcf().transFigure)
# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

THRESHOLD = 0.13

G3_corr = df_ohe.corr()["G3"]

df_ohe_after_drop_features = df_ohe.copy()

for key, value in G3_corr.items():  # Use .items() instead of .iteritems()
    if abs(value) < THRESHOLD:
        df_ohe_after_drop_features.drop(columns=key, inplace=True)


df_ohe_after_drop_features.drop(columns=["age"], axis=1, inplace=True)

# Calculate the correlation matrix for all columns
correlation_matrix = df_ohe_after_drop_features.corr()

# Extract the correlation values for the 'G3' column
correlation_with_G3 = correlation_matrix['G3']

# Create a heatmap of the correlation values
plt.figure(figsize=(5, 13))
sns.heatmap(correlation_with_G3.to_frame(), annot=True, cbar=True)
plt.show()

X = df_ohe_after_drop_features.drop('G3',axis = 1)
y = df_ohe_after_drop_features['G3']

df_ohe_after_drop_features.head()

X_all_features_except_G3 = df_ohe.drop('G3',axis = 1)
y_G3 = df_ohe ['G3']

def train_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

    model1 = LinearRegression()
    model2 = BayesianRidge()
    model3 = RandomForestRegressor()
    model4 = GradientBoostingRegressor()
    model5 = DecisionTreeRegressor()
    model6 = Ridge()
    model7 = Lasso()

    models = [model1, model2, model3, model4, model5, model6, model7 ]
    model_name_list = ['LinearRegression', 'BayesianRidge', 'RandomForestRegressor', 'GradientBoostingRegressor',
           'DecisionTreeRegressor', 'Ridge', 'Lasso']

    # Dataframe for results
    results = pd.DataFrame(columns=['MAE', 'RMSE', 'RMSE by cross validation', 'MSE', 'R^2'], index=model_name_list)
    for i, model in enumerate(models):
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_test_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_test_pred)

       # Cross-validation
        scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_squared_error', cv=5)
        rmse_cross_val = np.sqrt(-scores.mean())

        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse, rmse_cross_val ,mse,r_squared ]

    return results

train_regression_model(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

best_model1 = BayesianRidge()
best_model1.fit(X_train, y_train)

y_test_pred = best_model1.predict(X_test)

n,m=polyfit(y_test, y_test_pred, 1)
plt.figure(figsize=(8,5))
plt.scatter(x = y_test, y = y_test_pred, c="red")
plt.plot(y_test,  m*(y_test) + n   ,':', c="black")

plt.xlabel("Truth")
plt.ylabel("Predicted")

with open('reg_model.pkl', 'wb') as file:
    pickle.dump(best_model1, file)

train_regression_model(X_all_features_except_G3, y_G3)

X_train, X_test, y_train, y_test = train_test_split(X_all_features_except_G3, y_G3, test_size=0.1, shuffle=True, random_state=42)

best_model2 = BayesianRidge()
best_model2.fit(X_train, y_train)

y_test_pred2 = best_model2.predict(X_test)

n,m=polyfit(y_test,y_test_pred2,1)
plt.figure(figsize=(8,5))
plt.scatter(x = y_test, y = y_test_pred2, c="blue")
plt.plot(y_test,  m*(y_test) + n   ,':', c="black")
plt.xlabel("Truth")
plt.ylabel("Predicted")

X = df_ohe_after_drop_features.drop('G3',axis = 1)
y = df_ohe_after_drop_features['G3'].apply(lambda x: 'pass' if x >= 10 else 'fail')

def train_binary_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

    model1 = LogisticRegression()
    model2 = MultinomialNB()
    model3 = BaggingClassifier()
    model4 = DecisionTreeClassifier()
    model5 = LinearSVC()
    model6 = SGDClassifier()
    model7 = KNeighborsClassifier()
    model8 = RandomForestClassifier()
    model9 = GradientBoostingClassifier()

    models = [model1, model2, model3, model4, model5, model6, model7, model8, model9 ]
    model_name_list = ['LogisticRegression', 'MultinomialNB', 'BaggingClassifier', 'DecisionTreeClassifier',
           'LinearSVC', 'SGDClassifier', 'KNeighborsClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']
            # Dataframe for results
    results = pd.DataFrame(columns=["Test Accuracy", "Train Accuracy"], index=model_name_list)

    for i, model in enumerate(models):
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

       # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        accuracy_train = accuracy_score(y_train, y_train_pred)



        model_name = model_name_list[i]
        results.loc[model_name, :] = [accuracy, accuracy_train ]

    return results

train_binary_classification_model(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

y_test_pred_gb = gb_model.predict(X_test)

cm = confusion_matrix(y_test, y_test_pred_gb)

# Plot confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

bagg_model = BaggingClassifier()
bagg_model.fit(X_train, y_train)

y_test_pred_bgg = bagg_model.predict(X_test)

cm2 = confusion_matrix(y_test, y_test_pred_bgg)

# Plot confusion matrix as a heatmap
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

with open('binay_classification_model.pkl', 'wb') as file:
    pickle.dump(gb_model, file)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)

dot_data = StringIO()
tree.export_graphviz(decision_tree_model, out_file=dot_data, feature_names=X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# Get feature importances
feature_importances = decision_tree_model.feature_importances_
# Match feature importances with feature names (assuming you have feature names in X_train.columns)
feature_importance_dict = dict(zip(X_train.columns, feature_importances))

# Sort feature importances in descending order
sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

features, importances = zip(*sorted_feature_importances)

# Create a horizontal bar plot for feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(len(features)), importances, align='center')
plt.yticks(range(len(features)), features)
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Decision Tree Classifier')
plt.gca().invert_yaxis()  # Invert y-axis to display the most important features at the top
plt.show()

X_all_features_except_G3 = df_ohe.drop('G3',axis = 1)
y_G3 = df_ohe ['G3'].apply(lambda x: 'pass' if x >= 10 else 'fail')

train_binary_classification_model(X_all_features_except_G3, y_G3)





