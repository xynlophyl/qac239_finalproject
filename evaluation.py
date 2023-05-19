import pandas as pd

def evaluate(df_path, df_type):
    
    '''
    evaluate the model by aggregating results
    '''    

    br = '-----------------------'

    print(br, f'EVALUATION OF {df_type}',br, '\n')
    
    # loading results df
    df = pd.read_csv(df_path)

    df = df.drop(columns=df.columns[0])

    df = evaulate_accuracy(df)

    # overall accuracy
    correct_predictions = df['predict_result'].aggregate('sum')

    accuracy_pct = correct_predictions/len(df)*100

    accuracy_pct = round(accuracy_pct, 2)

    print(f'OVERALL ACCURACY OF MODEL: \n (total no. correct predictions) / (total no. faces) = {correct_predictions}  / {len(df)} = {accuracy_pct}% \n')
    
    # accuracy by age group
    df_group = df.groupby('age_range')

    count_group = df_group['file'].aggregate('count')

    correct_group = df_group['predict_result'].aggregate('sum')

    accuracy_group = (correct_group / count_group)*100

    accuracy_group = accuracy_group.apply(lambda x: round(x, 2))

    group_eval = pd.concat([correct_group, count_group, accuracy_group], axis=1)

    group_eval.columns = ['correct_predictions', 'total_count', 'accuracy']

    print('ACCURACY BY AGE GROUP \n', group_eval, '\n')

    # detection speed
    detection_speeds = df[['face_detection_time', 'age_detection_time', 'total_detection_time']]
    detection_speeds = detection_speeds.mean()
    detection_speeds = detection_speeds.apply(lambda x: round(x, 2))

    print(f'DETECTION SPEEDS: \n{detection_speeds}')

    print(br, 'END', br)


def evaulate_accuracy(df):

    '''
    evaluating accuracy of model predictions
    '''

    for index in range(len(df)):

        observation = df.iloc[index]

        actual_range = observation['age_range']
        predicted_range = observation['predicted_range'] 
        
        df.at[index, 'predict_result'] = actual_range == predicted_range

    return df

# pre test evaluation
# evaluate('outputs/predictions/pre_predictions.csv', 'pre_test')
# pre test adjusted evaluation
# evaluate('outputs/adjusted_pre_predictions.csv', 'adjusted_pre_test')