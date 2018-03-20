# TextClassification
Python ML code to classify texts within a csv file into given classification types.

import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
 
 
class TextClassification:
    def __init__(self,mode='prediction',model_list=None,document_loc=None,saved_model_loc=None):
        self.saved_model_loc = saved_model_loc
 
        if mode =='training':
            self.model_list = model_list       
                  
    def start_training(self): 
        df = pd.read_csv('TextFile_Test.csv', encoding = "ISO-8859-1") #encoding to handle certain ascii characterset
        df_x=df["NRRTV_TX"].astype('U') #convert the dtype object to unicode string
        df_y=df["DC_REASON_CD"].astype('U')
                                      
        for clf_name,clf in self.model_list.items():           
            '''--------- Training --------'''
            x_train,x_test,y_train,y_test= train_test_split(df_x,df_y,test_size=0.20)
            count_vect = CountVectorizer()
            X_train_counts = count_vect.fit_transform(x_train)
            tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
 
            clf.fit(X_train_tfidf, y_train)
           
            n=100000
            feature_names = count_vect.get_feature_names()
            coefs_with_fns = sorted(zip(clf.coef_[0], feature_names)) 
            top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
            with open("Feature_Set.csv", 'w', newline="\n", encoding="utf-8") as file:
                writer=csv.writer(file)
                for (coef_1, fn_1), (coef_2, fn_2) in top:
                    writer.writerow([str(fn_1)])
               
#           
#            '''--------- Testing ---------'''   
            docs_test = x_test            
            test_counts = count_vect.transform(docs_test)
            X_test_tfidf = tfidf_transformer.transform(test_counts)
            predicted = clf.predict(X_test_tfidf)        
           
            '''--------- Statistics ------'''   
            print('Classifier:',clf_name,' -> Accuracy Score:',
                  np.mean(predicted == y_test)*100,'%')
           
            '''--------- Persist the trained model ------'''
            if self.saved_model_loc:
                dictToPersist = {'vect':count_vect,
                                 'transformer':tfidf_transformer,
                                 'classifier':clf,
                                 'labels':list(y_train)}                   
                self._persist_model(dictToPersist,clf_name)
           
            print('-------- Training Completed !! ---------')
           
    def _persist_model(self,clf,name):
        joblib.dump(clf, self.saved_model_loc +'/'+name+'.pkl') # Save the trained ML Model
   
    def classify_my_text(self,model_name,text):       
        dictObj = joblib.load(self.saved_model_loc+'/'+model_name+'.pkl')
        count_vect = dictObj['vect']
        test_counts = count_vect.transform(text)
        tfidf_transformer = dictObj['transformer']
        transformed_text = tfidf_transformer.transform(test_counts)
        classifier = dictObj['classifier']
        predicted = classifier.decision_function(transformed_text)
        result = pd.DataFrame(predicted, columns=classifier.classes_)
        result1=result.transpose()
        sortedResult=result1.sort_values(0,ascending=False).head(5)
        top5=sortedResult.transpose()
        DC_REASON_CD_Predicted = top5.columns.values
        print (DC_REASON_CD_Predicted)
        print('\nPredicted Text Classification -> ', model_name, '\n')
        with open("Predicted_Classification.csv", 'w', newline="\n", encoding="utf-8") as file:
            writer=csv.writer(file)
            for i in top5.columns.values:
                print (i)
                writer.writerow([str(i)])
              
 
def main():   
    """
    Step-1> Train the model. 
    Step-2> Predict classification for any given text using SGDClassifier.
    """
   
    #@TODO- Update these two paths to local
    saved_models = <local path>
   
    ##### STEP-1 - TRAINING ########       
    print ("Invoking Train Classification")
    classifiers={           
        'SGDClassifier': SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)       
    }   
    TextClassification(mode='training',model_list=classifiers,
                       saved_model_loc=saved_models).start_training()
 
    #### STEP-2 - CLASSIFICATION PREDICTION ########
    TextClassification(saved_model_loc=saved_models)\
                      .classify_my_text('SGDClassifier',['testing classification sample'])
 
   
#--- Start the script ---#
if __name__ == '__main__':
    main()

