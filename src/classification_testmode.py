from utils.general_utils import *
from utils.globals import *
  
def main():
  """
  -----------------------------------Data Preprocessing---------------------------------------
  """
  globals.init_result()
  dataset = pd.read_csv("../dataset/TTSWING.csv")
  print( dataset.info() )

  # experimental settings
  experiment_num = 10
  label_name = 'testmode' # change data from here
  save_filename = "classification_{0}_result.xlsx".format(label_name)
  
  dataset = dataset.drop(['teststage', 'id', 'date', 'fileindex', 'count', 'age', 'gender', 'play years', 'height', 'weight', 'handedness', 'hold racket handed'], axis = 1)
  subdata = dataset.copy() # change data from here

  for i in range( experiment_num ):
    # shuffle and spilt data
    x_train, x_test, y_train, y_test = shuffle_and_spilt( subdata, label_name )
    # -----------------------------------DNN---------------------------------------
    DNN_multi( x_train, x_test, y_train, y_test, 3 ) 
    
    # -----------------------------------Decision Tree---------------------------------------
    Decision_Tree( x_train, x_test, y_train, y_test )
    
    # -----------------------------------K-nearest neighbor---------------------------------------
    K_nearest_neighbor( x_train, x_test, y_train, y_test )
    
    # -----------------------------------Support Vector Machine---------------------------------------
    Support_Vector_Machine( x_train, x_test, y_train, y_test )
    
    # -----------------------------------Random Forest---------------------------------------
    Random_Forest( x_train, x_test, y_train, y_test )
    
    # -----------------------------------Logistic Regression---------------------------------------
    Logistic_Regression_multi( x_train, x_test, y_train, y_test )
    
    # -----------------------------------Naive Bayes---------------------------------------
    Naive_Bayes( x_train, x_test, y_train, y_test )
    
  # end for

  DNN_binary_result = globals.DNN_binary_result
  DNN_multi_result = globals.DNN_multi_result
  dtc_result = globals.dtc_result
  knn_result = globals.knn_result
  svc_result = globals.svc_result 
  rfc_result = globals.rfc_result 
  lm_result = globals.lm_result
  gnb_result = globals.gnb_result
  
  result_stat( [ ["DNN multi", DNN_multi_result], [ "Decision", dtc_result ], [ "knn", knn_result ], [ "svm", svc_result ],
               [ "rf", rfc_result ], [ "lg", lm_result ], [ "nb", gnb_result ] ], save_filename )
  
# end main()
               
if __name__ == "__main__":
  main()
# end if
