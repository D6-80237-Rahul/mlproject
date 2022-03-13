import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import plotly.figure_factory as ff
import numpy as np


def ml_operation():
    st.title(" Execute ML Operation")
    st.success('Model Training Complete')
    ok = st.button("Press for Predictions")
    if ok:
        
        

                 df2 = pd.get_dummies(df1, drop_first=True)
                 url = 'https://drive.google.com/file/d/1xDHWoHulIn8SAVKfLW_ybixNyg62e42G/view?usp=sharing'
                 path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
                 dft = pd.read_csv(path)

            
          
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
      
                 cv = GridSearchCV(knn, param_grid=parameters,cv=kfold,verbose=3)
                 model=cv.fit( X_train , y_train )
                 y_pred1 = model.predict(X_test)
                 a=accuracy_score(y_test, y_pred1)
                 st.caption('The predictions are based on K Nearest Neighbour ML model .')
                 st.caption('Accuracy Score:')
                 st.text(a)           
                
          
                 ypred = model.predict(df2)

                 ypred[ypred<0] = 0

                 pred = le.inverse_transform(ypred)
                 dfp = pd.DataFrame(pred)
                 result = pd.concat([df1, dfp], axis=1, join='outer')
                 st.title("Predictions")
                 st.write(result)
                 


                 x1 = np.result[2]
                 x2 = np.result[3]
                 
                 x3 = np.result[5]
# Group data together
                 hist_data = [x1, x2, x3]

                 group_labels = ['X', 'Y','Activity']

# Create distplot with custom bin_size
                 fig = ff.create_distplot(
                     hist_data, group_labels, bin_size=[.1, .25, .5,])

# Plot!
                 st.plotly_chart(fig, use_container_width=True)
                 


                 csv = result.to_csv().encode('utf-8')
                 st.download_button(
                 "Download Prediciton(.csv)",
                 csv,
                 "file.csv",
                 "text/csv",
                 key='download-csv'
                 )    



    
st.sidebar.title("PG-DBDA Capston Project")
st.sidebar.markdown("Project Guide:Ms.Manasi Yeole")
st.sidebar.markdown("Develpoed by:Avinash\nPrakhar\nRahul\nShivam\nMayuresh")

st.title("Human Activity Recognititon")
st.title("Test File")
uploaded_file2 = st.file_uploader("Choose a test file")
if uploaded_file2 is not None:
  
    df1 = pd.read_csv(uploaded_file2)
    st.write(df1)

    ls = ml_operation()
    
    

