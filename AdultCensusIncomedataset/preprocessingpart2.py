import pandas as pd
import sys
import numpy as np
def readCSV(file):
    df=pd.read_csv(file,sep=",|\s+",header=None)
    return df
def dropMissingValues(dataframe):
    df=dataframe.replace(' ?',np.nan)
    df1=df.dropna(axis=0,how='any')
    output=df1[df1.columns[-1]]
    features=df1.iloc[:,:-1]
    return output,features
def preprocessFeatures(features):
    features1=features.copy()
    for col in features:
        if features[col].dtype!=object:
            mean=features[col].mean()
            std=features[col].std()
            features1[col]=(features[col]-mean)/std
        else:
            features1=pd.concat([features1,pd.get_dummies(features[col])],axis=1)
            
    return features1
def preprocessOutput(output):
    output1=output.copy()
    #for col in output:
    if output.dtype==object or output.dtype==bool:
        array=output.unique()
        l1=list(array)
        output.replace(l1,[i for i in range(len(l1))],inplace=True)
    
    elif output.dtype!=object:
        output=output<output.mean()
        output=output.astype(int)
        return output
        
    
    return output
def main():
    print("enter the name of the csv file to preprocess")
    file=sys.stdin.readline().strip()
    dataframe=readCSV(file)
    output,features=dropMissingValues(dataframe)
    
    features1=preprocessFeatures(features)
    output1=preprocessOutput(output)
    
    df1=pd.concat([features1,output1],axis=1)
    df2=df1.select_dtypes(include=[np.number])
    print(df2)
    print("enter the name of the csv file")
    filename=sys.stdin.readline().strip()
    df2.to_csv(filename,header=False,index=False)
main()
