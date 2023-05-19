import helper
from model import Model
import evaluation
import plot

def main():
    
    '''
    main function
    '''

    BASE_PATH = './assets/'

    # set up dataset
    df = helper.create_df(BASE_PATH)

    print(df)

    # split df into training and testing sets
    train, pre_test, post_test = helper.random_sampling(df)

    # print(df.shape, train.shape, pre_test.shape, post_test.shape)
    # print(df.shape[0]/df.shape[0], train.shape[0]/df.shape[0], pre_test.shape[0]/df.shape[0], post_test.shape[0]/df.shape[0])

    # plot ages distribution (before data pre-processing)
    # plot.plot_distribution(df, label = 'age')

    # plot age group distribution (after data pre-processing)
    # plot.plot_distribution(df, label = 'age_range')

    # initialize model with pre-trained parameters
    model = Model(
        path = BASE_PATH,
        age_model_mean = (78.4263377603, 87.7689143744, 114.895847746),
        age_weights = "models/gad/age_deploy.prototxt",
        age_config = "models/gad/age_net.caffemodel"
        )

    # run model on pre-test set
    pre_prediction = model.test(pre_test, output_file='pre_predictions.csv') 
    # post_prediction = model.test(post_test, output_file='post_predictions.csv')   
    
    # downloading csv (long runtime)
    pre_prediction.to_csv('./outputs/pre_predictions.csv')


    # evaluate pre-test predictions
    pre_eval = evaluation.evaluate(pre_prediction)

    # train model using training set
    new_model_mean, new_weights, new_config = model.train(train)

    # initialize new model
    updated_model = Model(
        path = BASE_PATH,
        age_model_mean = new_model_mean,
        age_weights = new_weights,
        age_config= new_config
        )

    # run model on post_test set
    old_post_prediction = model.test(pre_test)
    old_post_prediction.to_csv('./outputs/old_post_prediction.csv')

    updated_post_prediction = updated_model.test(post_test)
    updated_post_prediction.to_csv('./outputs/updated_post_prediction.csv')


    # evaluate post_test set
    old_post_evaluate = evaluation.evaluate(old_post_prediction)
    updated_post_evaluate = evaluation.evaluate(updated_post_prediction)

if __name__ == '__main__':
    
    main()
# main()

