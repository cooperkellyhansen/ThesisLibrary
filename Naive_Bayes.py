
import pandas as pd
from collections import Counter as Counter# Counter to help with label voting


def minkowski_distance(a, b, p):
    # Store the number of dimensions
    dim = len(a)

    # Set initial distance to 0
    distance = 0

    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d]) ** p

    distance = distance ** (1 / p)

    return distance



def knn_predict(X_train, X_test, y_train, k, p):

    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []
        #print('test point', test_point)

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)

        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'])
        #print('distances', df_dists)

        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
        #print('k closest Distances', df_nn)

        # Create counter object to track the labels of k closest neighbors
        # im having to switch from list to dataframe here because the
        # I have to index out the dist values
        nearest_FIP = [y_train[i] for i in df_nn.index.values]
        #nearest_FIP = [item for sublist in nearest_FIP for item in sublist] #flatten list
        #print('nearest FIPs', nearest_FIP)

        #lets try averaging the FIPs for the nearest neighbors
        df_nn = df_nn.values.tolist()
        prediction = sum(nearest_FIP)/len(nearest_FIP)
        for i in range(len(df_nn)):
            if df_nn[i] == 0.0:
                prediction = nearest_FIP[i]
                break

        # Append prediction to output list
        y_hat_test.append(prediction)

    return y_hat_test
