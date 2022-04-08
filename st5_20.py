import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


#Fuction for model training and predctions


def ml_operation():
    st.title(" Execute ML Operation")
    
   
    ok = st.button("Click for Predictions")
    if ok:
              with st.spinner('Please wait application is retrieving trainig dataset from Google drive....'):       
                  url = 'https://drive.google.com/file/d/1DaZ9BktNdprOezWaWRP43w6XCTDf5K7d/view?usp=sharing'
                  path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
                  df = pd.read_csv(path)
                  st.success('Trainig dataset retrived')
#preprocessing
              with st.spinner('Please wait model training on progress'):     
                 df['Arrival_Time'] = df['Arrival_Time'].astype('datetime64[ns]')
                 df['year'] = df['Arrival_Time'].dt.year
                 df['month'] = df['Arrival_Time'].dt.month
                 df['day'] = df['Arrival_Time'].dt.day
                 df['hour'] = df['Arrival_Time'].dt.hour
                 df['minute'] = df['Arrival_Time'].dt.minute
                 df['second'] = df['Arrival_Time'].dt.second
                 df.drop(columns=['Arrival_Time','model','device','index'],inplace=True)
                 
                 #Specifyimg Features and Target 
                 x = df.iloc[:,0:11]
                 x.drop(columns=['activity'],inplace=True)
                 y = df.iloc[: , -7]
                 le = LabelEncoder()
                 le_y = le.fit_transform(y)
                 
                 #Train Test split
                 X_train,X_test,y_train,y_test = train_test_split(x,le_y,test_size=0.3,
                                                          stratify=le_y,random_state=2022)

#Model Training - K Nearest Neighbour.

                 parameters = {'n_neighbors': [1,5]}
       
                 kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)
                 knn = KNeighborsClassifier()
      
                 cv = GridSearchCV(knn, param_grid=parameters,cv=kfold,verbose=3)
                 model=cv.fit( X_train , y_train )
                 y_pred1 = model.predict(X_test)
                 a=accuracy_score(y_test, y_pred1)
                 st.subheader('The predictions are based on K Nearest Neighbour ML model .')
                 st.subheader('Accuracy Score:')
                 st.text(a)  
                 st.success('Model Training Complete')
                 
                 #Preprocesing Test set
              with st.spinner('Please wait model fitting on uploaded data is on progress.....'):    
                 df=df2
                 df['Arrival_Time'] = df['Arrival_Time'].astype('datetime64[ns]')
                 df['year'] = df['Arrival_Time'].dt.year
                 df['month'] = df['Arrival_Time'].dt.month
                 df['day'] = df['Arrival_Time'].dt.day
                 df['hour'] = df['Arrival_Time'].dt.hour
                 df['minute'] = df['Arrival_Time'].dt.minute
                 df['second'] = df['Arrival_Time'].dt.second
                 df.drop(columns=['Arrival_Time'],inplace=True)
                 ypred = cv.predict(df)
                 pred = le.inverse_transform(ypred)


                 dfp = pd.DataFrame(pred)
                 dfp=dfp.dropna()
                 dfp.columns = ['Predicted_Activity']


                 cols=["year","month","day"]
                 df2['Date'] = df2[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
                 cols1=["hour","minute","second"]
                 df2['Time'] = df2[cols1].apply(lambda y: '-'.join(y.values.astype(str)), axis="columns")
                 df2.drop(columns=['year','month','day','hour','minute','second'],inplace=True)

                 result = pd.concat([df2, dfp], axis=1, join='outer')
                 result=result.dropna()  
                 st.success('Activites predicted successfully')


                 st.title("Charts based on Predictions")
                
                 
                 
                 #Pie chart
                 
                 st.subheader('Pie Chart')
                 
                 a=result['Predicted_Activity'].value_counts()
                 b=result.Predicted_Activity.unique()
                 
                 x = np.array(a)
                 mylabels = b
                 fig = plt.figure(figsize=(20, 15))
                 plt.pie(x, labels = mylabels,autopct='%1.0f%%',textprops={'fontsize': 35},shadow=True)
                 fig
                 with st.expander("See explanation"):
                        st.write("""
    The chart above shows the types of activites use has done  
""")
                 
                
                 


                 #Bar graph
                 st.subheader('Bar Chart')
                 Time = result['Time']
                 Activity = result['Predicted_Activity']


                 fig = plt.figure(figsize = (40, 30))
                  
                 # creating the bar plot
                 plt.bar(Activity,Time, color ='blue',
                         width = 0.4)
                 plt.rcParams.update({'font.size': 50})
                 fig
                 with st.expander("See explanation"):
                     st.write("""The Chart above shows the types of activities user has done""")
# st.write("""The chart above shows the types of activites use has done""")
                 st.title("Predictions in Tabular Form ")                 
                 st.write(result)
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

st.title("Human Activity Recognition")
st.caption("Upload file to get Predicted Activities(containing following fields- arrival_time,x,y,z{co-ordinates}) - in .csv Format")
uploaded_file2 = st.file_uploader("Browse file in local meomory")
if uploaded_file2 is not None:
  
    df2 = pd.read_csv(uploaded_file2)
    st.write(df2)
    genre = st.sidebar.radio("Data Analysis",('none','Charts'))
    st.subheader('Select Data Analysis Option from Sidebar')
    if genre=='none':
        st.subheader('Do you want to proceed,without graphs?!')
        ok1 = st.button("Proceed")
        if ok1:
                    with st.spinner('Please wait Model training on progress....'):
                        gauth = GoogleAuth()
                        gauth.LocalWebserverAuth()
                        drive=GoogleDrive(gauth)
                        file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1zrxEmECbKh5Wt9LUApKMiRkt79UJV2Hh')}).GetList()
                        url='https://drive.google.com/file/d/'
                        for file in file_list:
                            a=file['title']
                            url=url+file['id']+'/view?usp=sharing'
                     #url = 'https://drive.google.com/file/d/1uwk4mG30Dkd0FI9DB7gRt3cfSS6qiowR/view?usp=sharing'
                    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
                    df = pd.read_csv(path)
    #preprocessing
                    df['Arrival_Time'] = df['Arrival_Time'].astype('datetime64[ns]')
                    df['year'] = df['Arrival_Time'].dt.year
                    df['month'] = df['Arrival_Time'].dt.month
                    df['day'] = df['Arrival_Time'].dt.day
                    df['hour'] = df['Arrival_Time'].dt.hour
                    df['minute'] = df['Arrival_Time'].dt.minute
                    df['second'] = df['Arrival_Time'].dt.second
                    df.drop(columns=['Arrival_Time','model','device','index'],inplace=True)

                     #Specifyimg Features and Target 
                    x = df.iloc[:,0:11]
                    x.drop(columns=['activity'],inplace=True)
                    y = df.iloc[: , -7]
                    le = LabelEncoder()
                    le_y = le.fit_transform(y)
                     
                     #Train Test split
                    X_train,X_test,y_train,y_test = train_test_split(x,le_y,test_size=0.3,
                                                             stratify=le_y,random_state=2022)

    #Model Training - K Nearest Neighbour.

                    parameters = {'n_neighbors': [1,3,5,6,7,9,11,13,15]}
           
                    kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)
                    knn = KNeighborsClassifier()
          
                    cv = GridSearchCV(knn, param_grid=parameters,cv=kfold,verbose=3)
                    model=cv.fit( X_train , y_train )
                    y_pred1 = model.predict(X_test)
                    a=accuracy_score(y_test, y_pred1)
                    st.caption('The predictions are based on K Nearest Neighbour ML model .')
                    st.caption('Accuracy Score:')
                    st.text(a)
                     
                    

                    df=df2
                    df['Arrival_Time'] = df['Arrival_Time'].astype('datetime64[ns]')
                    df['year'] = df['Arrival_Time'].dt.year
                    df['month'] = df['Arrival_Time'].dt.month
                    df['day'] = df['Arrival_Time'].dt.day
                    df['hour'] = df['Arrival_Time'].dt.hour
                    df['minute'] = df['Arrival_Time'].dt.minute
                    df['second'] = df['Arrival_Time'].dt.second
                    df.drop(columns=['Arrival_Time'],inplace=True)
                    ypred = cv.predict(df)
                    pred = le.inverse_transform(ypred)


                    dfp = pd.DataFrame(pred)
                    dfp=dfp.dropna()
                    dfp.columns = ['Predicted_Activity']


                    cols=["year","month","day"]
                    df2['Date'] = df2[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
                    cols1=["hour","minute","second"]
                    df2['Time'] = df2[cols1].apply(lambda y: '-'.join(y.values.astype(str)), axis="columns")
                    df2.drop(columns=['year','month','day','hour','minute','second'],inplace=True)

                    result = pd.concat([df2, dfp], axis=1, join='outer')
                    result=result.dropna()



                    st.title("Predictions")
                    st.write(result)
                 
                     
        
   
    else:
       
            ml_operation()
            
