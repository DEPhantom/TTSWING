import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.utils import *
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import openpyxl
from tqdm import trange
import time
from utils import globals

def evaluate(y_test, predicted):
  print(metrics.classification_report(y_test,predicted))
  print(metrics.confusion_matrix(y_test, predicted))
  accuracy = accuracy_score(y_test, predicted)
  print("Accuracy: %.2f%%" % (accuracy * 100.0))
  precision = metrics.precision_score(y_test, predicted,average='macro')
  print( 'precision:',precision ) 
  recall = metrics.recall_score(y_test, predicted,average='macro')
  print( 'recall:',recall )
  print('F1-score:',metrics.f1_score(y_test,predicted,average='macro'))
  
  return [ float("%.2f" % (accuracy*100.0)), precision, recall ]
  
# end evaluate()
  
def shuffle_and_spilt( input_data, label_name ) :
  shuffle_data = shuffle(input_data) # shuffle, input is dataframe

  data = shuffle_data.drop(label_name, axis = 1) # spilt data and label
  label = shuffle_data[[label_name]]

  # Normalization 
  minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
  data = minmax_scale.fit_transform(data)

  # Change type to numpy array 
  data = np.array( data )
  label = np.array( label )
  
  x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=0, train_size = 0.8)
  return x_train, x_test, y_train, y_test
  
# end shuffle_and_spilt()

def age_encoding(data):
  # Age 13~15 -> class 0
  # Age 16~18 -> class 1
  # Age 19~22 -> class 2
  # Age 23~28 -> class 3
  bins = [13,15,18,22,28]
  labels=[0,1,2,3]
  data['age'] = pd.cut(data['age'], bins=bins, labels=labels, include_lowest=True)
  
# end age_encoding()

def result_stat( total_result, filename ):

  workbook = openpyxl.Workbook()  
  sheet = workbook.active
  
  title = [""]
  col_data = []
  mean_list = []
  std_list = []
  
  # [["DNN", DNN_result], .... ]
  
  for loc in range( len( total_result ) ):
    title = title+[ total_result[loc][0], "", ""]
  # end for
  
  sheet.append( title )
  
  # DNN_result [ [93.91, 0.9385, 0.9285], ..... ] # accuracy, precision, recall
  for i in range( len( total_result[0][1] ) ):
    row_data = [ i+1 ]
    for method_count in range( len( total_result ) ):
      row_data = row_data + total_result[method_count][1][i]
    # end for
    
    sheet.append( row_data )
    row_data = []
    
  # end for
  
  # sheet.extend( [[],[],[]] ) #to be tried
  sheet.append( [] )
  sheet.append( [] )
  sheet.append( [] )

  for i in range( len( total_result )*3 ):
    for expriment_count in range( len( total_result[0][1] ) ):
      col_data.append( sheet[expriment_count+2][i+1].value )
    # end for
    
    mean_list.append( np.mean( col_data ) )
    std_list.append( np.std( col_data ) )
    col_data = []
  # end for

  sheet.append( ["average"]+mean_list )
  sheet.append( ["standard deviation"]+std_list )
  

  workbook.save(filename)

# end def

def DNN_binary( x_train, x_test, y_train, y_test ) :
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(input_dim=x_train.shape[1], units=70))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=100))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=100))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=10))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=1))
  model.add(tf.keras.layers.Activation('sigmoid'))
  model.summary()
  
  #train the model
  model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['acc'])
  train_history = model.fit(x_train, y_train, validation_split=0.2, batch_size=200, epochs=20)
  
  predicted = model.predict(x_test)
  predicted = np.where(predicted>0.5,1,0)
  print('DNN binary')
  globals.DNN_binary_result.append( evaluate(y_test, predicted) )
  
# end DNN_binary()

def DNN_multi( x_train, x_test, y_train, y_test, classes_num ) :
  
  y_train = to_categorical(y_train, classes_num ) # one hot encode
  
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(input_dim=x_train.shape[1], units=70))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=100))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=100))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=10))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=classes_num))
  model.add(tf.keras.layers.Activation('softmax'))
  model.summary()
  
  #train the model
  model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
  train_history = model.fit(x_train, y_train, validation_split=0.2, batch_size=200, epochs=20)
  
  predicted = model.predict(x_test)
  predicted = np.where(predicted>0.5,1,0)
  predicted = np.argmax(predicted,axis=1) # decode
  globals.DNN_multi_result.append( evaluate(y_test, predicted) )
  
# end DNN_multi()

def Decision_Tree( x_train, x_test, y_train, y_test ) :
  dtc = DecisionTreeClassifier()
  dtc.fit(x_train, y_train)
  dt_predicted = dtc.predict(x_test)
  globals.dtc_result.append( evaluate(y_test, dt_predicted) )
  
# end Decision_Tree()

def K_nearest_neighbor( x_train, x_test, y_train, y_test ) :
  n = 0
  acc = 0
  for i in trange(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    knn_predicted = knn.predict(x_test)
    if acc < accuracy_score(y_test, knn_predicted):
        n = i
        acc = accuracy_score(y_test, knn_predicted)
    # end if
    time.sleep(0.2)

  # end for


  knn = KNeighborsClassifier(n_neighbors=n)
  knn.fit(x_train, y_train)
  knn_predicted = knn.predict(x_test)
  print('k = {0} is picked.'.format(n))
  globals.knn_result.append( evaluate(y_test, knn_predicted) )
  
# end K_nearest_neighbor()

def Support_Vector_Machine( x_train, x_test, y_train, y_test ) :
  svc = SVC()
  svc.fit(x_train, y_train)
  svm_predicted = svc.predict(x_test)
  globals.svc_result.append( evaluate(y_test, svm_predicted) )
  
# end Support_Vector_Machine()

def Random_Forest( x_train, x_test, y_train, y_test ) :
  rfc=RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =50, min_samples_leaf = 10)
  rfc.fit(x_train, y_train)
  pred = rfc.predict(x_test)
  rfc_predicted = rfc.predict(x_test)
  globals.rfc_result.append( evaluate(y_test, rfc_predicted) )
  
# end Random_Forest()

def Linear_Regression( x_train, x_test, y_train, y_test ) :
  lm = LinearRegression()
  lm.fit(x_train, y_train)
  lm_predicted = lm.predict(x_test)
  lm_predicted = np.where(lm_predicted>=0.5,1,0)
  globals.lm_result.append( evaluate(y_test, lm_predicted) )
  
# end Linear_Regression()

def Logistic_Regression( x_train, x_test, y_train, y_test ) :
  lm = LogisticRegression()
  lm.fit(x_train, y_train)
  lm_predicted = lm.predict(x_test)
  globals.lm_result.append( evaluate(y_test, lm_predicted) )
  
# end Logistic_Regression()

def Logistic_Regression_multi( x_train, x_test, y_train, y_test ) :
  lm = LogisticRegression(multi_class='multinomial')
  lm.fit(x_train, y_train)
  lm_predicted = lm.predict(x_test)
  print('coef: {0}\nintercept: {1}'.format(lm.coef_, lm.intercept_))
  globals.lm_result.append( evaluate(y_test, lm_predicted) )
  
# end Logistic_Regression_multi()

def Naive_Bayes( x_train, x_test, y_train, y_test ) :
  gnb = GaussianNB()
  gnb.fit(x_train, y_train)
  nb_predicted = gnb.predict(x_test)
  globals.gnb_result.append( evaluate(y_test, nb_predicted) )
  
# end Naive_Bayes()
