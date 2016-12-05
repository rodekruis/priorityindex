# Script developed by Marco Velliscig (marco.velliscig AT gmail.com)
# for the NL Red Cross
# released under GNU GENERAL PUBLIC LICENSE Version 3 for the NL Red Cross
# This script is for illustrative reasons only.
# If you are interested in using the script please contact the 510.global team
# 



# notes of this version
# automatic selection of best feautures/ combination of feautures
# grid parameter search
# neural network



def learn_damage(dict_instance, show_plots = False):


    #-000000000000000000000000000000      OPTIONS

    # it drops from the training matrix regions with 0 windspeed and 0 rainfall
    # this can be set to false if it helps to learn the 0 damage regions (problem1)
    # to fix the total damaged houses
    drop_0wind_and_0rain= False
    drop_0wind= False
    drop_0damage= True #only way to predict over haima
    drop_50damage=False
    use_log = dict_instance['alg_use_log']

    dict_instance['alg_date']=datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")

    #show_plots = False
    #fix problem2 Rainfallme is an object rather than float
    fix_Rainfallme = True

    typhoon_to_predict=dict_instance['typhoon_to_predict']






    predict_on = dict_instance['alg_predict_on']
    learn_matrix = dict_instance['learn_matrix']


    if 'perc' in predict_on.split('_'): use_perc = True
    else: use_perc=False
    #print summary


    #rrrrrrrrrrrrrrrrrrrrrrrrrr READ DATA

    #typhoon_sequence = 
    df_ty = pd.read_csv(learn_matrix)

    if typhoon_to_predict != '' :
        indexes_to_pred =  (df_ty['typhoon_name'] == typhoon_to_predict)
        df_ty_to_pred = df_ty.loc[indexes_to_pred]
        indexes_to_learn =  (df_ty['typhoon_name'] != typhoon_to_predict)
        df_ty = df_ty.loc[indexes_to_learn]
    #explore type of columns
    #for i in df_ty.columns : print i , df_ty[i].dtype

    if fix_Rainfallme:
        df_ty['Rainfallme'] = pd.to_numeric(df_ty['Rainfallme'],errors='coerce')
        df_ty['Rainfallme'] = df_ty['Rainfallme'].replace({np.nan:0})
    if typhoon_to_predict != '' :
        df_ty_to_pred['Rainfallme'] = pd.to_numeric(df_ty_to_pred['Rainfallme'],errors='coerce')
        df_ty_to_pred['Rainfallme'] = df_ty_to_pred['Rainfallme'].replace({np.nan:0})

        
    indexes_wind_no_rain = (df_ty.average_speed_mph > 0.0) & (df_ty.Rainfallme == 0.0)
    indexes_rain_no_wind = (df_ty.average_speed_mph == 0.0) & (df_ty.Rainfallme > 0.0)

    #print 'null values rain',sum(df_ty.Rainfallme.isnull())
    #print 'null values wind',sum(df_ty.average_speed_mph.isnull())
    #print 'municipalities with wind but no rain' , sum(indexes_wind_no_rain )
    #print 'municipalities with rain but no wind' , sum(indexes_rain_no_wind )


    if drop_0wind_and_0rain:
        indexes =  (df_ty.average_speed_mph > 0.0) | (df_ty.Rainfallme > 0.0)
        df_ty = df_ty.loc[indexes]
    if drop_0wind:
        indexes =  (df_ty.average_speed_mph > 0.0)
        df_ty = df_ty.loc[indexes]    
    if drop_0damage:
        indexes =  (df_ty['pop_15'] > 0.0)
        df_ty = df_ty.loc[indexes]
    if drop_50damage:
        indexes =  (df_ty['pop_15'] > 50.0)
        df_ty = df_ty.loc[indexes]

 

    # merger now done in pipeline.py
    df = df_ty


    #$$$$$$$ create dataframe for prediction
    df_to_pred = df_ty_to_pred 









    # fill the NaN values in the length of the coast with 0 (the actual length)
    df['coast_length']=df['coast_length'].fillna(0)
    #recompute the coastal-perimeter ratio
    df['cp_ratio'] = df['coast_length'] / df['perimeter']



    # $$$$$$$ create dataframe for prediction
    df_to_pred['coast_length']=df_to_pred['coast_length'].fillna(0)
    #recompute the coastal-perimeter ratio
    df_to_pred['cp_ratio'] = df_to_pred['coast_length'] / df_to_pred['perimeter']





    # TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT   TRANSFOR FEATURES

    df['poverty_frac'] = (df['poverty_perc']+1)/100.


    #add one to make it log convertible
    df['cp_ratio'] = df['cp_ratio']+1
    df['coast_length'] = df['coast_length']+1
    df['average_speed_mph'] = df['average_speed_mph']+1
    df['distance_typhoon_km']=df['distance_typhoon_km']+1
    df['Rainfallme'] = df['Rainfallme']+1





    #normalize the coordinates
    df['x_pos'] = (df['x_pos'] -df['x_pos'].min()) / (df['x_pos'].max() -df['x_pos'].min())
    df['y_pos'] = (df['y_pos'] -df['y_pos'].min()) /( df['y_pos'].max() -df['y_pos'].min())



    # $$$$$$$ create dataframe for prediction
    df_to_pred['poverty_frac'] = (df_to_pred['poverty_perc']+1)/100.

    #add one to make it log convertible
    df_to_pred['cp_ratio'] = df_to_pred['cp_ratio']+1
    df_to_pred['coast_length'] = df_to_pred['coast_length']+1
    df_to_pred['average_speed_mph'] = df_to_pred['average_speed_mph']+1
    df_to_pred['distance_typhoon_km']=df_to_pred['distance_typhoon_km']+1
    df_to_pred['Rainfallme'] = df_to_pred['Rainfallme']+1



    df_to_pred['x_pos'] = (df_to_pred['x_pos'] -df_to_pred['x_pos'].min()) / (df_to_pred['x_pos'].max() -df_to_pred['x_pos'].min())
    df_to_pred['y_pos'] = (df_to_pred['y_pos'] -df_to_pred['y_pos'].min()) /( df_to_pred['y_pos'].max() -df_to_pred['y_pos'].min())



    transformation = 'log_div_max_log'
    #print  'transform as ' , transformation , '\n'
    name_old =['area_km2', 
               'pop_15',
               'pop_density_15',
               'average_speed_mph',
               'coast_length',
               'poverty_frac',
               'distance_typhoon_km',
               'Rainfallme',
               'cp_ratio',
               'distance_first_impact',
               'proj_distance_first_impact',               
               'mean_elevation_m',
               'mean_ruggedness',
               'ruggedness_stdev',
               'mean_slope',
               'slope_stdev']


    name_new=  ['area_log', 
                'pop15_log', 
                'pop_density_log',
                'wind_speed_log',
                'coastline_length',
                'poverty_frac_log',
                'dist_path_log',
                'rainfallme_log',
                'cp_ratio_log',
                'distance_first_impact_log',
                'proj_distance_first_impact_log',               
                'elevation_log',
                'ruggedness_mean_log',
                'ruggedness_stdv_log',
                'slope_mean_log',
                'slope_stdv_log']

    #print  'varaiables to transform ' ,  '\n' , zip(name_old , name_new)
    apply_transform(df ,name_old ,name_new, trans = transformation)
    apply_transform(df_to_pred ,name_old ,name_new, trans = transformation)

    #df[name_new].hist()
    #plt.show()

    df.loc[df.poverty_frac_log.isnull()]=df.poverty_frac_log.mean()


    df_to_pred.loc[df_to_pred.poverty_frac_log.isnull()]=df_to_pred.poverty_frac_log.mean()



    learn_variables=dict_instance['learn_variables']



    if show_plots :  
        df[learn_variables].hist()
        plt.show()


    X = df[learn_variables]
    ids = df['Mun_Code']
    typhoon = df['typhoon_name']
    if use_perc :
        normalization = df['pop_15']/400.
    else:
        normalization = 1.

    #consider missin values as zeros    
    df[predict_on] = df[predict_on].replace({np.nan:0.0})

    if use_log :
        Y = np.log(df[predict_on])
    else:
        Y = (df[predict_on])




    if show_plots :    
        Y.hist()
        plt.show()
    #print Y.describe()



    hyperparams_dict ={}
    hyperparams_dict['GBT'] = {"loss" : ['ls', 'lad', 'huber'],
                               "n_estimators" : [100,500,700,1000],
                               "learning_rate": [0.01, 0.05, 0.1, 0.2],
                               "max_depth": [3, 2,None],
                               'max_features': ['auto', 'sqrt', 'log2'],
                               "min_samples_split": sp_randint(1, 11),
                               "min_samples_leaf": sp_randint(1, 11),
                               "criterion": ["mse", "friedman_mse"]}


    hyperparams_dict['randomforest']= {"n_estimators" : [10,50,100,500],
                                       "max_depth": [3, 2,None],
                                       'max_features': ['auto', 'sqrt', 'log2'],
                                       "min_samples_split": sp_randint(1, 11),
                                       "min_samples_leaf": sp_randint(1, 11),
                                       "n_jobs":[-1],
                                       "bootstrap": [True, False],
                                       "criterion": ["mse"]}

    hyperparams_dict['linreg'] = {"alpha" : 10.0 ** np.arange(-4, 1, 0.2),
                                  "l1_ratio": np.arange(0. , 1. , 0.1),
                                  "fit_intercept":[False,True]}

    hyperparams_dict['NN']= {"hidden_layer_sizes":[(5, ),(10, ),(5,3 ),(10,3),(200, 10),(500, 100,20)],
                             "activation" : ['identity', 'logistic', 'tanh', 'relu'],
                             "alpha" : 10.0 ** -np.arange(1, 7),

                             "max_iter" :  [int(10**x) for x in np.arange(2, 4,0.5)]}

    models_vector = [ dict_instance['alg_model']]


    for model_name in models_vector:

    #====================

        hyperparams = hyperparams_dict[model_name]

        

        Y_pred , best = model_pred(dict_instance,X,Y,hyperparams , model_type=model_name , n_iter_search = 40, n_cv_sets = 10 )

        if use_log :
            Y_n_houses = np.exp(Y)
            Y_pred_n_houses = np.exp(Y_pred)
        else:
            Y_n_houses = (Y)
            Y_pred_n_houses = (Y_pred)  



        if use_perc :
            Y_n_houses = Y_n_houses*normalization
            Y_pred_n_houses = Y_pred_n_houses*normalization
            




        mean_error_num_houses =np.mean(abs((Y_pred_n_houses) - (Y_n_houses)))
        median_error_num_houses = np.median(abs((Y_pred_n_houses) - (Y_n_houses)))
        std_error_num_houses = np.std(abs((Y_pred_n_houses) - (Y_n_houses)))
        print model_name , ' mean house damage  error' ,mean_error_num_houses , 'stdev ' ,std_error_num_houses
        print model_name , ' median house damage  error' , median_error_num_houses

        dict_instance['val_mean_error_num_houses']=mean_error_num_houses
        dict_instance['val_median_error_num_houses']=median_error_num_houses
        dict_instance['val_std_error_num_houses'] =std_error_num_houses
        #dict_instance['_error_num_houses']        

    #====================








        predict_on_tag = predict_on
        Y_df_h = pd.DataFrame((Y_n_houses) )



        Y_df_h.columns = ['num_'+predict_on_tag+'_true']


        if use_log :
            Y_df_h[predict_on_tag+'_true'] = np.exp(Y)
            Y_df_h[predict_on_tag+'_pred'] = np.exp(Y_pred)
        else:
            Y_df_h[predict_on_tag+'_true'] = (Y)
            Y_df_h[predict_on_tag+'_pred'] = (Y_pred)



        
        Y_df_h['typhoon_name']=typhoon

        Y_df_h['num_'+predict_on_tag+'_pred']   = Y_pred_n_houses
        Y_df_h['M_Code'] = df['Mun_Code']
        Y_df_h['Municipality'] = df['admin_L3_name']

        Y_df_h['pop']    = normalization
        Y_df_h['perc_'+predict_on_tag+'_pred'] = Y_df_h['num_'+predict_on_tag+'_pred']/normalization
        Y_df_h['perc_'+predict_on_tag+'_true'] = Y_df_h['num_'+predict_on_tag+'_true']/normalization


        Y_df_h['num_'+predict_on_tag+'_error'] = abs(Y_df_h['num_'+predict_on_tag+'_pred'] - Y_df_h['num_'+predict_on_tag+'_true'])
        Y_df_h['num_'+predict_on_tag+'_error_noabs'] = (Y_df_h['num_'+predict_on_tag+'_pred'] - Y_df_h['num_'+predict_on_tag+'_true'])

        Y_df_h['perc_'+predict_on_tag+'_error'] = (Y_df_h['perc_'+predict_on_tag+'_pred'] - Y_df_h['perc_'+predict_on_tag+'_true'])

        Y_df_h['rel_num_'+predict_on_tag+'_error'] = Y_df_h['num_'+predict_on_tag+'_error'] / Y_df_h['num_'+predict_on_tag+'_true'] 

        Y_df_h_sort = Y_df_h.sort_values('perc_'+predict_on_tag+'_true', axis=0)
        Y_df_h_sort = Y_df_h_sort.reset_index()
        if show_plots :
            Y_df_h_sort['perc_'+predict_on_tag+'_true'].plot()
            Y_df_h_sort['perc_'+predict_on_tag+'_pred'].plot()
            Y_df_h_sort['perc_'+predict_on_tag+'_error'].plot()
            plt.show()

            Y_df_h_sort['num_'+predict_on_tag+'_error_noabs'].hist(bins =30)
            plt.show()
        accuracy , perc_up , perc_down = compute_accuracy(Y_df_h ,'perc_'+predict_on_tag+'_true', 'perc_'+predict_on_tag+'_pred',[0,10,20,30,40,50,60,70,80,90,100,200])
        print  accuracy , perc_up , perc_down
        dict_instance['val_accuracy'] = accuracy
        dict_instance['val_perc_up'] = perc_up
        dict_instance['val_perc_down'] = perc_down     
        # X.loc[Y_df_h['num_'+predict_on_tag+'_error'] > 10000]

        # for col in learn_variables:
        #     col_zscore = col + '_zscore'
        #     df[col_zscore] = (df[col] - df[col].mean())/df[col].std()
        #     print df.loc[Y_df_h['num_'+predict_on_tag+'_error'] > 10000,col_zscore]
        #     print '===', np.mean(np.abs(df.loc[Y_df_h['num_'+predict_on_tag+'_error'] > 10000,col_zscore]))

        Y_df_h.to_csv('./output/'+ dict_instance['run_name']+ "_pred_" + model_name + ".csv")


        if typhoon_to_predict != '' :
        
            with open('./output_pickle/'+dict_instance['run_name']+'model'+model_name, 'rb') as f:
                print f
                model = cPickle.load(f)

                ### Use model for prediction
                print str(model)






                #Use the best model to predict damage from new typhoon

                if use_perc :
                    normalization = df_to_pred['pop_15']/400.
                else:
                    normalization = 1.




                X_to_pred = df_to_pred[learn_variables]
                print X_to_pred.describe()
                Y_to_pred = model.predict(X_to_pred)

                if use_log :
                    Y_to_pred_n_houses = np.exp(Y_to_pred)
                else:
                    Y_to_pred_n_houses = (Y_to_pred)  



                if use_perc :
                    Y_to_pred_n_houses = Y_to_pred_n_houses*normalization




                Y_df_h_to_pred = pd.DataFrame((Y_to_pred_n_houses) )
                Y_df_h_to_pred.columns = ['num_'+predict_on_tag+'_pred']

                if use_log :
                    Y_df_h_to_pred[predict_on_tag+'_pred'] = np.exp(Y_to_pred)
                else:
                    Y_df_h_to_pred[predict_on_tag+'_pred'] = (Y_to_pred)


                Y_df_h_to_pred['M_Code'] = df_to_pred['Mun_Code']
                Y_df_h_to_pred['Municipality'] = df_to_pred['admin_L3_name']
                Y_df_h_to_pred['typhoon_name']=typhoon_to_predict
                Y_df_h_to_pred['perc_'+predict_on_tag+'_pred'] = Y_df_h_to_pred['num_'+predict_on_tag+'_pred']/normalization


                Y_df_h_to_pred.to_csv('./output_unseen_predictions/'+ dict_instance['run_name']+ "_unseen_pred_" + model_name + typhoon_to_predict +".csv")








def compute_accuracy(df_to_pred,tag_pred ,tag_true, categories):
    
    df_to_pred[tag_pred+'_category'] = pd.cut(df_to_pred[tag_pred],categories )
    df_to_pred[tag_true +'_category'] = pd.cut(df_to_pred[tag_true],categories )
    df_comp = df_to_pred[df_to_pred[tag_true].notnull()]

    print 'number of nan ' , len(df_to_pred) -  len(df_comp)
    
    perc_up = float(sum(df_comp[tag_true+'_category'] < df_comp[tag_pred+'_category'])) / len(df_comp[tag_true+'_category'])

    perc_down =float(sum(df_comp[tag_true+'_category'] > df_comp[tag_pred+'_category'])) / len(df_comp[tag_true+'_category'])

    accuracy = float(sum(df_comp[tag_true+'_category'] == df_comp[tag_pred+'_category'])) / len(df_comp[tag_true+'_category'])
    print 'accuracy' , accuracy , 'perc_up' , perc_up, 'perc_down', perc_down
    return  accuracy , perc_up , perc_down






    for name in typhoon_list :
        df.loc[df.typhoon_name == name , 'distance'] = np.sqrt((df.loc[df.typhoon_name == name , 'x_pos'] - start[name][0])**2 +(df.loc[df.typhoon_name == name , 'y_pos'] - start[name][1])**2  )


# Utility function to report best scores
def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(


            score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print 'cross validation scores', score.cv_validation_scores
        print("Parameters: {0}".format(score.parameters))
        print("")



def norm_column(df , name_vec , new_name_vec , norm_type = '0mean1std', rename = False):
    """ Normalize column across differnt train and test dataset  """
    #the min should be the min of the min to avoid negative params
    #the max should be the max of the max to avoid bigger than one params
    for name,  new_name in zip(name_vec, new_name_vec):
        min =df[name].min() 
        max = df[name].max() 

        if norm_type == 'm_min_div_minmax':
            df[new_name] = (df[name] - min )/ (max-min)
        if norm_type == 'div_max':
            df[new_name] = (df[name])/ (max)  
      
        #print df[new_name].describe()


def apply_transform(df , name_vec , new_name_vec, trans = 'log', drop = False, rename = False):
    """ Normalize column across differnt train and test dataset  """
    #the min should be the min of the min to avoid negative params
    #the max should be the max of the max to avoid bigger than one params
    for name,  new_name in zip(name_vec, new_name_vec):

        temp = df[name]
        #if trans.split('_')[0] == 'log' : temp = temp.replace({0:1})

        if trans == 'log' : df[new_name] = np.log10(temp)
        if trans == 'log_p1' : df[new_name] = np.log10(temp+1)
        if trans == 'log_div_max_log' : df[new_name] = np.log10(temp) / np.log10(df[name].max()) 

        if trans == 'div_max' : df[new_name] = (temp) /    temp.max()
        #print df[new_name].describe()
        if drop : df = df.drop([name], axis=1)




def fill_null(df_vec,name_vec , groupby_col = 'all' , method = 'fill_median' , sample = 'first', drop = False, rename = False):
    """ fill missing values using only the train dataset to compute medians for each group """
    if method == 'fill_median':


        if sample == 'first' : 
            df_sample = df_vec[0]
        else:
            df_sample =  df_vec[0].append(df_vec[1])



        if groupby_col == 'all' : 
            med = df_sample.median()
        else : 
            med = df_sample.groupby(groupby_col).median()
        
        #print med

        for df in df_vec :
            for name in name_vec:
                if  rename == True :
                    new_name = name+method+groupby_col
                    df[new_name] = df[name]
                else:
                    new_name=name
                unique_values = (df[groupby_col].unique())
                for i in range(len(unique_values)) : 
                    df.loc[ (df[name].isnull())&(df[groupby_col]==unique_values[i]), new_name] = med[name].iloc[i] 
                if drop : df = df.drop([name], axis=1)


def one_hot(df , name_vec , drop = False) :
    """transform one column in one hot encoding
    possible to drop the original column if wanted"""


    for name in name_vec:
        class_one_hot =pd.get_dummies(df[name])

        #dat1.join(dat2)
        df = df.join( class_one_hot)
        #df_vec[i] = pd.concat([df_vec[i], class_one_hot], axis=1)
        if drop : df = df.drop([name], axis=1)
        #print df.columns
    return df

def model_pred(dict_instance,
               X,Y,
               hyperparams ,
               maximize='accuracy' ,
               model_type='logreg' ,
               n_iter_search = 30,
               n_cv_sets = 10 ,
               limits = [-3 , 1 , 0.5] ):



    X_train, X_test , Y_train , Y_test= train_test_split(X,Y, test_size = 0.3)

    param_dist = hyperparams
    
    if model_type=='linreg' :
        model = linear_model.ElasticNet()
    elif model_type=='lasso' :
        model = linear_model.Lasso()
    elif model_type=='randomforest' :
        model = RandomForestRegressor()
    elif model_type=='GBT' :
        model = GradientBoostingRegressor()
    elif model_type == 'NN':
        model = MLPRegressor()

    #how to decide the score
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,n_iter=n_iter_search, cv = n_cv_sets, )#scoring=
    #scorer(estimator, X, y)

    start = time()
    random_search.fit(X_train, Y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)

    score = sorted(random_search.grid_scores_, key=itemgetter(1), reverse=True)[0]
    dict_instance['val_best_score']= score.mean_validation_score
    dict_instance['val_stdev_best_score']= np.std(score.cv_validation_scores)
    
    #random_search.fit(X_train, Y_train)
    Y_test_pred= random_search.predict(X_test)

    
    print 'score test set' , random_search.score(X_test, Y_test)

    dict_instance['val_score_test'] = random_search.score(X_test, Y_test)


    print '===params best model' ,random_search.best_params_ 

    dict_instance['best_params'] = random_search.best_params_ 

    print ' diff pred and ground ' , np.mean(abs(Y_test_pred -Y_test ))
    


    model =random_search.best_estimator_
    
    Y_sub_pred=model_selection.cross_val_predict(model, X, y=Y, cv=n_cv_sets, n_jobs=2)
    #Y_sub_pred=(model.predict(X))
    best_score=random_search.best_score_ 
    print ' score on train set' , best_score

    if model_type=='linreg' :
        fi = zip(X_train.columns , model.coef_)
    elif model_type=='lasso' :
        model = linear_model.Lasso()
    elif model_type=='randomforest' :
        fi = zip(X_train.columns , model.feature_importances_)
    elif model_type=='GBT' :
        fi = zip(X_train.columns , model.feature_importances_)         
    elif model_type == 'NN':
        fi = [(0,0)]
    fi.sort(key=operator.itemgetter(1),reverse=True)
    for i in fi :
        #print i
        if model_type != 'NN': dict_instance['feat_'+i[0]] = i[1]
    
    Y_sub_pred=model_selection.cross_val_predict(model, X, y=Y, cv=n_cv_sets, n_jobs=2)

    model.fit(X, Y )

    with open('./output_pickle/'+dict_instance['run_name'] +'model'+model_type, 'wb') as f:
        cPickle.dump(model, f)




        
    return Y_sub_pred , best_score








#from __future__ import print_function


import operator
import cPickle
import numpy as np
from scipy import optimize
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier 
import csv as csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import string
from time import time 
from scipy.stats import randint as sp_randint
from operator import itemgetter
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
import datetime
import json
import os.path
now = datetime.datetime.now()

current_time  = now.strftime("%d/%m/%Y-%H:%M:%S")

version = '1p3'

run_file_log = 'runs_log_new_format'

typhoon_to_predict= 'new_typhoon'

#PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP      PROBLEMS
#problem0: is it possible to distinguish from east and west municipalities?
#problem1: if the predictor is run over the whole philippines can it predict 0 damage?
#problem2: Rainfallme is object



filename =  '/data/Dropbox/Documents_graafstroom/Data_science/Red_cross/Typhoon/pipeline_typhoon/5typhoons_v1.csv'

learn_variables = ['area_log', 
                   'pop15_log',
                   'x_pos',
                   'y_pos',
                   'pop_density_log',
                   'wind_speed_log',
                   'coastline_length',
                   'dist_path_log',
                   'poverty_frac_log',
                   'rainfallme_log',
                   'cp_ratio_log',
                   'proj_distance_first_impact_log',
                   'distance_first_impact_log',                   
                   'elevation_log',
                   'ruggedness_mean_log',
                   'ruggedness_stdv_log',
                   'slope_mean_log',
                   'slope_stdv_log',
                   'roof_iron',
                   'roof_conc',
                   'roof_half_conc',
                   'roof_wood',
                   'roof_straw',
                   'roof_makeshift',
                   'wall_conc',
                   'wall_wood',
                   'wall_half_wood',
                   'wall_iron',
                   'wall_bamboo',
                   'wall_makeshift']

learn_variables_nobuild = ['area_log', 
                   'pop15_log',
                   'x_pos',
                   'y_pos',
                   'pop_density_log',
                   'wind_speed_log',
                   'coastline_length',
                   'dist_path_log',
                   'poverty_frac_log',
                   'rainfallme_log',
                   'cp_ratio_log',
                   'proj_distance_first_impact_log',
                   'distance_first_impact_log',                   
                   'elevation_log',
                   'ruggedness_mean_log',
                   'ruggedness_stdv_log',
                   'slope_mean_log',
                   'slope_stdv_log',]

learn_variables_eventonly = [
                   'x_pos',
                   'y_pos',
                   'wind_speed_log',
                   'dist_path_log',
                   'poverty_frac_log',
                   'rainfallme_log',
                   'proj_distance_first_impact_log',
                   'distance_first_impact_log',   
]





dict_learn={
'test':{'alg_predict_on':'total_damage_houses_0p25weight_perc',
        'learn_matrix': filename,
        'code_version' : version,
        'code_date' : now.strftime("%d/%m/%Y-%H:%M:%S"),
        'alg_model' : 'linreg',
        'learn_variables' : learn_variables,
        'typhoon_to_predict' : typhoon_to_predict,
        'alg_use_log':True,},
    



    
    }
#learn variable
#mean error
#std
#percentiles
#r2 plus error
#accuracy
#best model

explore = True

if explore : 




    learn_variables_list = [learn_variables_nobuild , learn_variables , learn_variables_eventonly]
    var_list = ['part_damage_houses_perc','comp_damage_houses_perc','total_damage_houses_perc','total_damage_houses_0p25weight_perc']
    log_list = [False, True]
    model_list =  ['linreg','randomforest','GBT','NN']



    index_instance = 0 
    for variable in var_list:
        for log in log_list:
            for model in model_list:
                for learn_var in learn_variables_list :                

                    dict_learn['run_full_' + str(index_instance)]={'alg_predict_on':variable,
                                                         'learn_matrix': filename,
                                                         'code_version' : version,
                                                         'code_date' : now.strftime("%d/%m/%Y-%H:%M:%S"),
                                                         'alg_model' : model,
                                                         'learn_variables' : learn_var,
                                                         'typhoon_to_predict' : typhoon_to_predict,
                                                         'alg_use_log':log}
                    index_instance+=1




#then add it as a row
index = 0
for instance in dict_learn.keys() :
    dict_learn[instance]['run_name'] = instance
    print instance ,index +1 ,  'of ' , len(dict_learn)
    print dict_learn[instance]['alg_model'] ,  dict_learn[instance]['alg_predict_on'] ,  dict_learn[instance]['alg_use_log']
    
    learn_damage(dict_learn[instance])


    print dict_learn[instance] ,index +1 ,  'of ' , len(dict_learn)

    #col_to_df = [ dict_learn[instance]['learn_variables'],'predict_on' , 'version' , 'learn_matrix','date', 'model' ,   'use_log' , 'best_score' ,'stdev_best_score' ,'score_test',  'mean_error_num_houses','median_error_num_houses','std_error_num_houses']
    #df_runs= pd.DataFrame.from_dict(dict_learn[instance])
    

    with open(run_file_log+".json", 'a') as f:
        json.dump(dict_learn[instance], f)

    dict = dict_learn[instance].copy()
    dict.pop('best_params', None)
    dict.pop('learn_variables', None)    
    dict_row = { datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S") : dict}
    df_run= pd.DataFrame.from_dict(dict_row, orient='index')
    df_run.index.names = ['date']
    if index == 0 :
        
        if os.path.isfile(run_file_log+'.csv') :
            df_runs = pd.read_csv(run_file_log+'.csv',  index_col= 'date')
        else:
            df_run.to_csv(run_file_log+'.csv', index ='date')
            df_runs = pd.read_csv(run_file_log+'.csv',  index_col= 'date')
            
    df_runs=df_runs.append(df_run)

    index +=1

df_runs.to_csv(run_file_log+'.csv', index ='date')
df_runs.groupby('alg_model').describe()
df_runs.groupby('alg_predict_on').describe()
