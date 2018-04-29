import numpy as np
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics, cross_validation
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

label_encoder = LabelEncoder()

bugsDF = pd.read_csv("data/TFS_ML_Project_v4.csv", sep=",", index_col=0)
#print(bugsDF)

# Collect Valid Rows
def IsValidRow(x):
    return np.logical_and (
        np.logical_or (
            x['Train Data'] == 1,
            x['Test Data'] == 1
        ),
        x['To Delete'] != 1
    )

bugsDF = bugsDF[lambda x: IsValidRow(x)]
#print(bugsDF)

# Fill up Iteration Team and Iteration Release
def getVersion(iterPath):
    if(iterPath.find("Pro Services IS") >= 0): return "v5"
    if(iterPath.find("v4") >= 0): return "v4"
    if(iterPath.find("v5") >= 0): return "v5"
    if(iterPath.find("v6") >= 0): return "v6"
    return "vUnknown"

bugsDF['Iteration Team'] = bugsDF['Iteration Path'].apply(lambda x: x.split('\\')[1])
bugsDF['Iteration Release'] = bugsDF['Iteration Path'].apply(lambda x: getVersion(x))
#print(bugsDF)

inputDF = bugsDF.copy()
# 1 How Found
cleanup_nums = {'How Found':{np.nan: 0,
                                     "Developer Testing" : 1,
                                     "Manual Testing" : 2,
                                     "UAT" : 3,
                                     "Production" : 4}}
inputDF.replace(cleanup_nums, inplace=True)

# 2 Internal Priority
cleanup_nums = {'Internal Priority':{np.nan: 0,
                                     "4- Not required in Release" : 1,
                                     "3- If time Include in Release" : 2,
                                     "2- Before Final Release" : 3,
                                     "1- Blocker ASAP" : 4}}
inputDF.replace(cleanup_nums, inplace=True)


# 3 Reported By Client
cleanup_nums = {'Reported By Client':{np.nan: 0,
                                     "No" : 0,
                                     "Yes" : 1}}
inputDF.replace(cleanup_nums, inplace=True)

# 3b Severity
cleanup_nums = {'Severity':{np.nan: 0,
                                     "5- Cosmetic" : 1,
                                     "4- Low/Minor" : 2,
                                     "3- Medium/Moderate" : 3,
                                     "2- High/Serious": 4,
                                     "1- Critical/Fatal" : 5}}
inputDF.replace(cleanup_nums, inplace=True)

# 4 LabelEncode Iteration Team and Iteration Release
inputDF['Iteration Team'] = label_encoder.fit_transform(bugsDF['Iteration Team'])
inputDF['Iteration Release'] = label_encoder.fit_transform(bugsDF['Iteration Release'])

# 5 LabelEncode Node Name
inputDF['Node Name'] = label_encoder.fit_transform(bugsDF['Node Name'].fillna('Blank'))

# 6 LabelEncode Target Clients
inputDF['Target Clients'] = label_encoder.fit_transform(bugsDF['Target Clients'].fillna('Blank'))

# 7 LabelEncode Root Cause
inputDF['Root Cause'] = label_encoder.fit_transform(bugsDF['Root Cause'].fillna('Blank'))

# Save this result
pd.DataFrame.to_csv(bugsDF, "data/bugs_raw.csv")
pd.DataFrame.to_csv(inputDF, "data/bugs_LabelEncoded.csv")
#print(bugsDF)
#print(inputDF)

trainDF = inputDF[np.logical_and(inputDF['Train Data'] == 1, inputDF['Test Data'] == 0)].drop(['Train Data', 'Test Data', 'To Delete'], axis=1)
testDF  = inputDF[np.logical_and(inputDF['Train Data'] == 0, inputDF['Test Data'] == 1)].drop(['Train Data', 'Test Data', 'To Delete'], axis=1)

# Assign Features & Targets
features = trainDF.loc[:, "How Found":"Target Clients"]
targets = trainDF["ResolutionDays"]
#print(features)
#print(targets)

def AddCorrelations(df):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()

    ax = sns.heatmap(corr,
                mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True,
                ax=ax
            )

    #ax.xaxis.tick_top()
    # OR
    plt.gca().invert_yaxis()

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

def AddScatter(df, feature, index):
    plt.subplot(4, 1, index)
    plt.title(feature + " Vs ResolutionDays")
    plt.ylabel("ResolutionDays")
    plt.xlabel(feature)
    plt.scatter(df[feature], targets)

AddCorrelations(trainDF)

plt.subplots(2,1)
AddScatter(trainDF, "Internal Priority", 1)
AddScatter(trainDF, "Severity", 2)
AddScatter(trainDF, "Target Clients", 3)
AddScatter(trainDF, "Root Cause", 4)
plt.show()


