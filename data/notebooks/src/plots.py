import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../../../data/students.csv")

#creating a histogram of the math scores
sns.histplot(df["math score"], kde=True)
plt.title("Distribution of math scores")
plt.show()

#creating a barplot
df["average score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
sns.barplot(
    x = "parental level of education",
    y = "average score",
    data = df
)
plt.xticks(rotation = 45)
plt.title("Parental Education vs Average Student Score")
plt.show()
