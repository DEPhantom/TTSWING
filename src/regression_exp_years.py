import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, Lasso, Ridge, ElasticNet
import openpyxl
from tqdm import trange
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from statistics import mean

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

  title = [""]
  
  for loc in range( len( total_result ) ):
    title = title+[ "mae", "mse", "r squared"]
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
  
  sheet.append( [] )
  sheet.append( [] )
  sheet.append( [] )

  for i in range( len( total_result )*3 ):
    for expriment_count in range( len( total_result[0][1] ) ):
      col_data.append( sheet[expriment_count+3][i+1].value )
    # end for
    
    mean_list.append( np.mean( col_data ) )
    std_list.append( np.std( col_data ) )
    col_data = []
  # end for

  sheet.append( ["average"]+mean_list )
  sheet.append( ["standard deviation"]+std_list )
  

  workbook.save(filename)

# end def

def cal_r_squared(predictions, y_true ):
    mean = np.mean( y_true )
    ss_res = 0.0
    ss_total = 0.0
    for i in range( len(predictions) ):
      ss_res = ss_res+(predictions[i]-y_true[i])** 2
      ss_total = ss_total+(mean-y_true[i])**2
    # end for

    return ( 1-ss_res/ss_total )[0]
# end cal_r_squared()

def cal_mse(predictions, y_true ):
    error = 0.0
    for i in range( len(predictions) ):
      error = error+(predictions[i]-y_true[i])**2
    # end for

    return ( error/len( predictions ) )[0]
# end cal_mse()

def cal_mae(predictions, y_true ):
    error = 0.0
    for i in range( len(predictions) ):
      error = error+np.abs(predictions[i]-y_true[i])
    # end for

    return ( error/len( predictions ) )[0]
# end cal_mae()

def plot_reg_tor(pred, label, method_name):
  acc_arr = []
  index = []
  t = 0.1
  while t < 5:
    index.append(t)
    count = 0
    for i in range(len(pred)):
      if np.abs(pred[i]-label[i]) < t:
        count+=1
    acc_arr.append(count/len(pred))
    t += 0.1

  plt.plot(index,acc_arr, label=method_name)
  plt.legend()

# plot_reg_tor()

def DNN_multi_Regressor( x_train, x_test, y_train, y_test) :
  global dnn_result
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(input_dim=x_train.shape[1], units=70, kernel_initializer='normal', activation="relu"))
  model.add(tf.keras.layers.Dense(units=100, kernel_initializer='normal', activation="relu"))
  model.add(tf.keras.layers.Dense(units=100, kernel_initializer='normal', activation="relu"))
  model.add(tf.keras.layers.Dense(units=10, kernel_initializer='normal', activation="relu"))
  model.add(tf.keras.layers.Dense(units=1, kernel_initializer='normal'))
  model.summary()

  #train the model

  model.compile(loss='mean_absolute_error', optimizer="adam", metrics=['mse', 'mae'])

  train_history = model.fit(x_train, y_train, validation_split=0.2, batch_size=200, epochs=200)
  
  pred = model.predict(x_test)
  dnn_result.append( [ cal_mae(pred, y_test), cal_mse(pred, y_test), cal_r_squared(pred, y_test) ] )
  plot_reg_tor(pred, y_test,'DNN')
  
# end DNN_multi_Regressor()

def DecisionTree_Regressor( x_train, x_test, y_train, y_test ) :
  global dtc_result
  dtc = DecisionTreeRegressor()
  dtc.fit(x_train, y_train)
  dt_predicted = dtc.predict(x_test)
  dtc_result.append( [ cal_mae(dt_predicted, y_test), cal_mse(dt_predicted, y_test), cal_r_squared(dt_predicted, y_test) ] )
  plot_reg_tor(dt_predicted, y_test,'Decision tree')
  return cal_mae(dt_predicted, y_test)
  
# end DecisionTree_Regressor()


def KNeighbors_Regressor( x_train, x_test, y_train, y_test ) :
  global knn_result
  best_r2 = float('-inf')
  best_k = 0
  for i in range(1,21):
    neigh = KNeighborsRegressor(n_neighbors=i)
    neigh.fit(x_train, y_train)
    neigh_predicted = neigh.predict(x_test)
    current_r2 = cal_r_squared(neigh_predicted, y_test)
    if best_r2 < current_r2:
      best_r2 = current_r2
      best_k = i

  neigh = KNeighborsRegressor(n_neighbors=best_k)
  neigh.fit(x_train, y_train)
  neigh_predicted = neigh.predict(x_test)
  current_r2 = cal_r_squared(neigh_predicted, y_test)
  print('best k = {0}'.format(best_k))
  knn_result.append( [ cal_mae(neigh_predicted, y_test), cal_mse(neigh_predicted, y_test), cal_r_squared(neigh_predicted, y_test) ] )
  plot_reg_tor(neigh_predicted, y_test,'KNN')

# end KNeighbors_Regressor()

def Support_Vector_Machine( x_train, x_test, y_train, y_test ) :
  global svr_result
  svr = SVR(kernel = 'rbf')
  svr.fit(x_train, y_train)
  svr_pred = svr.predict(x_test)
  svr_result.append( [ cal_mae(svr_pred, y_test), cal_mse(svr_pred, y_test), cal_r_squared(svr_pred, y_test) ] )
  plot_reg_tor(svr_pred, y_test,'SVM')

# end Support_Vector_Machine()

def Random_Forest( x_train, x_test, y_train, y_test ) :
  global rfg_result
  rfg=RandomForestRegressor(n_estimators=100,n_jobs = -1,random_state =50, min_samples_leaf = 10)
  rfg.fit(x_train, y_train)
  rfg_pred = rfg.predict(x_test)
  rfg_result.append( [ cal_mae(rfg_pred, y_test), cal_mse(rfg_pred, y_test), cal_r_squared(rfg_pred, y_test) ] )
  plot_reg_tor(rfg_pred, y_test,'RandomForest')

# end Random_Forest()

def Linear_Regression( x_train, x_test, y_train, y_test) :
  global lr_result
  model=LinearRegression()
  model.fit(x_train,y_train)
  lr_pred = model.predict(x_test)
  lr_result.append( [ cal_mae(lr_pred, y_test), cal_mse(lr_pred, y_test), cal_r_squared(lr_pred, y_test) ] )
  plot_reg_tor(lr_pred, y_test,'linear_regression')

# end Linear_Regression()

def Bayesian_Ridge( x_train, x_test, y_train, y_test ) :
  global br_result
  br = BayesianRidge()
  br.fit(x_train, y_train)
  br_predicted = br.predict(x_test)
  br_result.append( [ cal_mae(br_predicted, y_test), cal_mse(br_predicted, y_test), cal_r_squared(br_predicted, y_test) ] )
  plot_reg_tor(br_predicted, y_test,'Bayesian_Ridge')

# end Bayesian_Ridge()

dnn_result = []
lr_result = []
svr_result = []
rfg_result = []
knn_result = []
dtc_result = []
br_result = []

def main():
  global dnn_result
  global lr_result
  global svr_result
  global rfg_result
  global knn_result
  global dtc_result
  global br_result
  """
  -----------------------------------Data Preprocessing---------------------------------------
  """

  dataset = pd.read_csv("./dataset/TTSWING.csv")
  print( dataset.info() )

  # experimental settings
  testmode = 1
  experiment_num = 10
  label_name = 'play years' # change data from here
  save_filename = "regression_years_result_mode_{0}.xlsx".format(testmode)
  
  dataset = dataset.loc[dataset.testmode==testmode,]
  dataset = dataset.drop(['teststage', 'id', 'date', 'fileindex', 'count', 'height', 'weight', 'handedness', 'hold racket handed', 'testmode', 'gender'], axis = 1)
  subdata = dataset.copy() # change data from here

  for i in range( experiment_num ):
    # shuffle and spilt data
    x_train, x_test, y_train, y_test = shuffle_and_spilt( subdata, label_name )
    # # -----------------------------------DNN---------------------------------------
    DNN_multi_Regressor( x_train, x_test, y_train, y_test )

    # -----------------------------------Decision Tree---------------------------------------
    DecisionTree_Regressor( x_train, x_test, y_train, y_test )

    # # -----------------------------------K-nearest neighbor---------------------------------------
    KNeighbors_Regressor( x_train, x_test, y_train, y_test )

    # # -----------------------------------Support Vector Machine---------------------------------------
    Support_Vector_Machine( x_train, x_test, y_train, y_test )

    # # -----------------------------------Random Forest---------------------------------------
    Random_Forest( x_train, x_test, y_train, y_test )

    # # -----------------------------------Linear Regression---------------------------------------
    Linear_Regression( x_train, x_test, y_train, y_test )

    # # -----------------------------------Bayesian Ridge---------------------------------------
    Bayesian_Ridge( x_train, x_test, y_train, y_test )

  # end for

  result_stat( [ ["DNN multi", dnn_result], [ "Decision", dtc_result ], [ "knn", knn_result ], [ "svm", svr_result ],
               [ "rf", rfg_result ], [ "lr", lr_result ], [ "nb", br_result ] ], save_filename )
               
# end main()

if __name__ == "__main__":
  main()
# end if
