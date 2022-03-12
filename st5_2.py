import streamlit as st
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def ml_operation():
    st.title("ML operation")
    ok = st.button("Press for Predictions")
    if ok:
        
        

                 df2 = pd.get_dummies(df1, drop_first=True)
                 url = 'https://drive.google.com/file/d/1xDHWoHulIn8SAVKfLW_ybixNyg62e42G/view?usp=sharing?raw=true'

                 dft = pd.read_csv(url)
             
            
          
                 X = dft.iloc[:,0:5]
                 x = pd.get_dummies(X, drop_first=True)
                 y = dft.iloc[: , -1]
                 le = LabelEncoder()
                 le_y = le.fit_transform(y)
                 
                 X_train,X_test,y_train,y_test = train_test_split(x,le_y,test_size=0.3,
                                                               stratify=le_y,random_state=2022)
                 parameters = {'n_neighbors': [1,5]}
       
                 kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)
                 knn = KNeighborsClassifier()
       # TimeoutErroruned according to accuracy score
                 cv = GridSearchCV(knn, param_grid=parameters,cv=kfold,verbose=3)
                 model=cv.fit( X_train , y_train )
          
                 ypred = model.predict(df2)

                 ypred[ypred<0] = 0

                 pred = le.inverse_transform(ypred)
                 dfp = pd.DataFrame(pred)
                 result = pd.concat([df1, dfp], axis=1, join='outer')
                 st.title("Predictions")
                 st.write(result)

      
       

    
st.sidebar.title("This a Machine Learning Application:")
st.sidebar.markdown("Develpoed by:AVINASH,MAYURESH,PRAKHAR,RAHUL,SHIVAM")
st.title("ML Project")
st.title("Test File")
uploaded_file2 = st.file_uploader("Choose a test file")
if uploaded_file2 is not None:
  
    df1 = pd.read_csv(uploaded_file2)
    st.write(df1)

    ls = ml_operation()




