import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt



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
                 dfp.columns = ['Activity']
                 df2=df1.drop(columns='Creation_Time')
                 result = pd.concat([df2, dfp], axis=1, join='outer')
                 st.title("Predictions")
                 st.write(result)
                 


                 if genre == 'Bar_graph':
                         graph_df = result.groupby("Activity").mean()

                    
                   
                         bar_fig = plt.figure(figsize=(8,7))

                         bar_ax = bar_fig.add_subplot(111)
                     

                         sub_graph_df = graph_df[["Arrival_Time"]]

                         a=sub_graph_df.plot.bar(alpha=0.8, ax=bar_ax, title="time against activity");
                         st.text("This a plot to show most performed Activity")
                         bar_fig
                 elif genre == 'Pie_Chart' :
                         a=result['Activity'].value_counts()
                         b=result.Activity.unique()
                         x = np.array(a)
                         mylabels = b

                         fig = plt.figure(figsize=(10, 4))
                         plt.pie(x, labels = mylabels,autopct='%1.0f%%')

                         fig
                 elif genre == 'Scatter_Plot':
                     sx=result['Activity']
                     sy=result['Arrival_Time']
                     nsx=sx.to_numpy()
                     nsy=sy.to_numpy()

                     fig = plt.figure(figsize=(10, 4))
                     plt.scatter(nsx, nsy)

                       
                     fig
                      
                 else:
                         st.write("no graph")
                
                   
                 


    
st.sidebar.title("PG-DBDA Capston Project")
st.sidebar.markdown("Project Guide:Ms.Manasi Yeole")
st.sidebar.markdown("Develpoed by:Avinash\nPrakhar\nRahul\nShivam\nMayuresh")

st.title("Human Activity Recognititon")
st.title("Test File")
uploaded_file2 = st.file_uploader("Choose a test file")
if uploaded_file2 is not None:
  
    df1 = pd.read_csv(uploaded_file2)
    st.write(df1)
    genre = st.sidebar.radio("Select graph",('none','Bar_graph','Pie_Chart','Scatter_Plot'))
    st.subheader('Select the graph to analyse predictions from Sidebar')
    if genre=='none':
        st.subheader('No graph is choosen.DO you want to proceed?!')
        ok1 = st.button("Proceed")
        if ok1:
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
            dfp.columns = ['Activity']
            df2=df1.drop(columns='Creation_Time')
            result = pd.concat([df2, dfp], axis=1, join='outer')
            st.title("Predictions")
            st.write(result)
        
   
    else:
        ml_operation()
