# The script for feature_engineer
import pandas as pd
import pdb 



train_data = pd.read_csv('train.csv')
label_data = train_data[['Survived']]
test_data = pd.read_csv('test.csv')

#Step_1 dealing with the missing data
'''
目前遺失值有三種常見作法
1.用平均值、中值、分位數、眾數、隨機值等替代。缺點:等於人為增加了噪聲。

2.用其他變量做預測模型來算出缺失變量。缺點:有一個根本缺陷，如果其他變量和缺失變量無關，則預測的結果無意義。如果預測結果相當準確，則又說明這個變量是沒必要加入建模的。一般情況下，介於兩者之間。

3. 把變量映射到高維空間。比如性別，有男、女、缺失三種情況，則映射成3個變量：是否男、是否女、是否缺失。連續型變量也可以這樣處理。比如Google、百度的CTR預估模型，預處理時會把所有變量都這樣處理，達到幾億維。這樣做的好處是完整保留了原始數據的全部信息、不用考慮缺失值、不用考慮線性不可分之類的問題。缺點是計算量大大提升。而且只有在樣本量非常大的時候效果才好，否則會因為過於稀疏，效果很差。(這邊引用網路資料，方法三的例子應該只要兩個虛擬變數就可以了)
'''
def fillna_age(df):
    #Create the initial
    #for i in df:
    df['Initial'] = df.Name.str.extract('([A-Za-z]+)\.') 
    df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
    #filling the average age
    df.loc[(df.Age.isnull())&(df.Initial=='Mr'),'Age']=33
    df.loc[(df.Age.isnull())&(df.Initial=='Mrs'),'Age']=36
    df.loc[(df.Age.isnull())&(df.Initial=='Master'),'Age']=5
    df.loc[(df.Age.isnull())&(df.Initial=='Miss'),'Age']=22
    df.loc[(df.Age.isnull())&(df.Initial=='Other'),'Age']=46
    return df

def fillna_embark(df):
    df['Embarked'].fillna('S',inplace=True)
    return df

#執行補缺值    
#Step 2 製作特徵
#Create the family size feature
def family_size(df):
    df['family_size'] = 0
    df['family_size'] = df['Parch'] + df['SibSp']
    return df
#神奇的特徵
'''
titanic作為一個訓練項目，個人覺得練練手熟悉熟悉工具以及一些常見的特徵選擇和處理方法就行了，實在沒有太大的必要在這個任務上花太多精力去刷。好多成績非常好的人都是使用了一些非常tricky的技巧，實用性不是很大。比如，一般來說姓名對於預測能否生還沒有太大的價值，但在這個賽題的設置下，適當的考慮姓名可以發揮意想不到的作用。
The data indicates that woman and children were prioritized in rescue. Furthermore it appears that woman and children from the same family survived or perished together. There are 142 females and boys in the training dataset who have a relative (indicated by surname) in the training set that is a female or boy. Let's refer to these 142 passengers as members of "woman-child-groups".)
'''
#train_data['surename'] =  train_data.Name.apply(lambda x:str(x).split(',')[0])
#test_data['surename'] = test_data.Name.apply(lambda x: str(x).split(',')[0])
#average_df = train_data[['Survived','surename']].groupby(['surename']).mean()
#mapping_rule = {idx:rate for idx,rate in zip(average_df.index,average_df['Survived'])}
#def Name_feature(df):
#    df['surename'] =  df['surename'].map(mapping_rule)
#    df['surename'] = df['surename'].fillna(0)
#    return df
#Step3:變換變數
'''
我們可以觀察到像是性別等變數是字串資料，我們需要將字串資料轉換為數值資料。
'''
def transform(df):
    df['Sex'].replace(['male','female'],[0,1],inplace=True)
    df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
    return df

#Step4 選取特徵
def feature_selection(df):
    drop_cols = ['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Initial',]
    df.drop(drop_cols,axis=1,inplace=True)
    return df

def execute(df):
    df = fillna_age(df)
    df = fillna_embark(df)
    df = family_size(df)
#    df = Name_feature(df)
    df = transform(df)
    df = feature_selection(df)
    return df
    

train_data = execute(train_data)
test_data = execute(test_data)
train_data.drop(['Survived'],axis=1,inplace=True)
