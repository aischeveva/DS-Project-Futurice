import joblib
import numpy as np
import pandas as pd
import os, sys

def topic_classifier(): 
    #Load the model from file
    tight_model = joblib.load('random_forest_classifier_model.sav')
    tolerated_model = joblib.load('random_forest_classifier_5_diff_model.sav')
    #Change the directory to the csv file storage
    os.chdir("../web/source")
    for office in os.listdir():
        if (office != 'finalized_model.sav' and office != 'Topic Classifier.ipynb'
           and office != 'Random Forest.ipynb' and office != '.ipynb_checkpoints'
           and office != '.DS_Store' and office != 'classified_sectors'
           and office != 'Untitled.ipynb'):
            print('Classifying ' + office.replace(".csv",''))
            unseen_sector = pd.read_csv(office, error_bad_lines=False).drop(columns = "Unnamed: 0").T
            predictions = tight_model.predict(unseen_sector)
            #Find the lines which are unlikely to be trendy in the future
            index = []
            temp = 0
            for i in range(len(predictions)):
                if (predictions[i] == 0):
                    index.append(temp)
                    temp += 1
                else: 
                    temp += 1

            #Drop the line 
            sector = unseen_sector.T.drop(unseen_sector.T.columns[[index]], axis = 1).rename(index = {0: 2010, 1 : 2011, 2: 2012, 3: 2013, 4: 2014, 5: 2015, 6: 2016, 7: 2017, 8: 2018})
            if (sector.shape[1] == 0):
               unseen_sector = pd.read_csv(office, error_bad_lines=False).drop(columns = "Unnamed: 0").T
               predictions = tolerated_model.predict(unseen_sector)
            #Find the lines which are unlikely to be trendy in the future
               index = []
               temp = 0
               for i in range(len(predictions)):
                    if (predictions[i] == 0):
                        index.append(temp)
                        temp += 1
                    else: 
                        temp += 1

               unseen_sector = unseen_sector.T.drop(unseen_sector.T.columns[[index]], axis = 1).rename(index = {0: 2010, 1 : 2011, 2: 2012, 3: 2013, 4: 2014, 5: 2015, 6: 2016, 7: 2017, 8: 2018})
             #Save the reduced version to file 
               unseen_sector.to_csv(os.getcwd()[:-10] + '/web/source' + os.sep + office)
            
            else: 
               unseen_sector = unseen_sector.T.drop(unseen_sector.T.columns[[index]], axis = 1).rename(index = {0: 2010, 1 : 2011, 2: 2012, 3: 2013, 4: 2014, 5: 2015, 6: 2016, 7: 2017, 8: 2018})
                 #Save the reduced version to file 
               unseen_sector.to_csv(os.getcwd()[:-10] + '/web/source' + os.sep + office)

        else: 
            continue
