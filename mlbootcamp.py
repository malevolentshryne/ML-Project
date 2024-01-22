class Linear_Regression():

    def cost(self, X, W, b, y, lambda_):
        m = len(X)
        cost_ = np.sum(((np.dot(X, W)+b)-y)**2) + lambda_*np.sum((W**2))
        cost_ = cost_/(2*m)
        return cost_

    # def cost(X, W, b, y):
        #     cost_ = 0
        #     m = len(X)
        #     n = len(X[0])
        #     for i in range(m):
        #         cost_ += ((np.dot(X[i],W)+b)-y[i])**2
        #     cost_ = cost_/(2*m)
        #     return cost_

    def initialize_param(self, X):
        W = np.zeros(len(X[0]))
        b = 0
        return W, b

    # def gradient(X, W, b, y):
    #     m = len(X)
    #     n = len(X[0])
    #     dj_dw = np.zeros((n))
    #     dj_db = 0
    #     for j in range(n):
    #         dj_dw_j=0
    #         for i in range(m):
    #             dj_dw_j +=((np.dot(X[i],W)+b)-y[i])*X[i,j]
    #         dj_dw[j]=dj_dw_j
    #     for i in range(m):
    #         dj_db += ((np.dot(X[i],W)+b)-y[i])
    #     dj_dw = dj_dw/m
    #     dj_db = dj_db/m
    #     return dj_dw, dj_db

    def gradient_(self, X, W, b, y, lambda_):
        m = len(X)
        delta = (np.dot(X, W)+b)-y
        dj_dw = np.dot(X.T, delta) + lambda_*W
        dj_dw = dj_dw/m
        dj_db = np.sum(delta)
        dj_db = dj_db/m
        return dj_dw, dj_db

    def run_grad_descent(self, train_data, alpha, lambda_, iterations):
        X = train_data[:, 1:]
        y = train_data[:, 0]
        J_history= []
        X_norm = self.normalize(X)
        W_in, b_in = self.initialize_param(X)
        for i in range(iterations):
            J_history.append(self.cost(X_norm, W_in, b_in, y, lambda_))
            dj_dw, dj_db = self.gradient_(X_norm, W_in, b_in, y, lambda_)
            W_in = W_in - alpha*dj_dw
            b_in = b_in - alpha*dj_db
            if ((i+1)%10==0) :
                print(f"Cost at {i+1}th iteration is : {self.cost(X_norm, W_in, b_in, y, lambda_)}")
        return W_in, b_in, J_history

    def normalize(self,X):
        mu     = np.mean(X, axis=0)
        sigma  = np.std(X, axis=0)
        X_norm = (X - mu) / sigma
        return X_norm

    def normalize_test(self, X_train, X_cv):
        mu = np.mean(X_train, axis=0)
        sigma = np.std(X_train, axis=0)
        X = (X_cv - mu) / sigma
        return X

    # def rem_outliers(X, y):
    #     m = len(y)
    #     y_mean = np.mean(y)
    #     y_std = np.std(y)
    #     x = 0
    #     for i in range(m):
    #         if(((y[x]>y_mean+3*y_std)) or (y[x]<(y_mean -3*y_std))):
    #             y = np.delete(y, x, 0)
    #             X = np.delete(X, x, 0)
    #             x = x-1
    #         x = x+1
    #     return X, y

    def predict(self, X, X_train, W, b):
        #here X is already normalized
        X = self.normalize_test(X_train, X)
        m = len(X)
        y_pred = (np.dot(X, W)) + b
        return y_pred

    def R2_score(self, y_pred, y):
        SSR = np.sum((y_pred-y)**2)
        y_mean = np.mean(y)
        SST = np.sum((y-y_mean)**2)
        return 1- SSR/SST

class Polynomial_Regression():

    def cost(self, X, W, b, y):
        m = len(X)
        cost_ = np.mean(((np.dot(X,W)+b)-y)**2)
        cost_ = cost_/2
        return cost_

    #def cost(X, W, b, y):
    #  cost_ = 0
    #  m = len(X)
    #  n = len(X[0])
    #  for i in range(m):
    #     cost_ += ((np.dot(X[i],W)+b)-y[i])**2
    #  cost_ = cost_/(2*m)
    #  return cost_

    def gradient(self, X, W, b, y):
        m = len(X)
        delta = (X.dot(W)+b)-y
        dj_dw = (X.T.dot(delta))
        dj_dw = dj_dw/m
        dj_db = np.mean(delta)
        return dj_dw, dj_db

    # def gradient(X, W, b, y):
    #     m = len(X)
    #     n = len(X[0])
    #     dj_dw = np.zeros((n))
    #     dj_db = 0
    #     for j in range(n):
    #         dj_dw_j=0
    #         for i in range(m):
    #             dj_dw_j +=((np.dot(X[i],W)+b)-y[i])*X[i,j]
    #         dj_dw[j]=dj_dw_j
    #     for i in range(m):
    #         dj_db += ((np.dot(X[i],W)+b)-y[i])
    #     dj_dw = dj_dw/m
    #     dj_db = dj_db/m
    #     return dj_dw, dj_db

    def transform(self, X, degree):
        m = len(X)
        X_ = X.copy()
        X__ = X.copy()
        count = 0
        for i in range(0, degree+1):
            for j in range(0, degree-i+1):
                x_tem = (X_[:,0]**(i))*(X_[:,1]**(degree-i-j))*(X_[:,2]**(j))
                x_tem = np.reshape(x_tem,(m,1))
                X__ = np.concatenate((X__,x_tem), axis=1)
                count+=1
        for i in range(0, 3):
            X__ = np.delete(X__,0,1)
        X__ = np.reshape(X__,(m,count))
        return X__

    def run_grad_descent(self, X, W_in, b_in, y, alpha, iterations):
        for i in range(iterations):
            dj_dw, dj_db = self.gradient(X, W_in, b_in, y)
            W_in = W_in - alpha*dj_dw
            b_in = b_in - alpha*dj_db
        J = self.cost(X, W_in, b_in, y)
        return W_in, b_in, J

    def normalize(self, X):
        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        X_norm = (X - mean) / std_dev
        return X_norm

    def normalize_test(self, X_train, X_cv):
        mean = np.mean(X_train, axis=0)
        std_dev = np.std(X_train, axis=0)
        X = (X_cv - mean) / std_dev
        return X

    def predict(self, X, W, b):
        #here X is already normalized
        y_pred = X.dot(W)+b
        return y_pred

    # def predict(X, W, b):
    #   #here X is already normalized
    #   m = len(X)
    #   y_pred = np.zeros(m)
    #   for i in range(m):
    #       y_pred[i] = (np.dot(X[i], W)) + b
    #   return y_pred

    def generate_weight(self, X):
        n = len(X[0])
        W = np.zeros(n)
        return W

    def best_model(self, train_data, cv_data, degree, alpha, iterations):
        W = []
        b = []
        J = []
        X = train_data[:, 1:]
        y = train_data[:, 0]
        X_cv = cv_data[:, 1:]
        y_cv = cv_data[:, 0]
        X_ = X.copy()
        X_ = self.normalize(X_)
        for i in range (1, degree+1):
            if(i!=1):
                X_ = self.binomial(X, i)
                X_ = self.normalize(X_)
            W_in = self.generate_weight(X_)
            b_in = 0
            W_in, b_in, J_ = self.run_grad_descent(X_, W_in, b_in, y, alpha, iterations)
            W.append(W_in)
            b.append(b_in)
            J.append(J_)
            print(f"The final cost after {iterations} iterations for degree {i} is: {J_}")
            print(f"The R2_score on cv set for degree {i} is: {self.R2_score(self.predict(self.normalize_test(self.binomial(X, i), self.binomial(X_cv, i)), W_in, b_in), y_cv)}")
        return W, b, J

    def binomial(self, X, degree):
        X_ = X.copy()
        for i in range(2, degree+1):
            x_tem = self.transform(X, i)
            X_ = np.concatenate((X_, x_tem), axis=1)
        return X_

    def R2_score(self, y_pred, y):
        SSR = np.sum((y_pred-y)**2)
        y_mean = np.mean(y)
        SST = np.sum((y-y_mean)**2)
        return 1- (SSR/SST)

    def train_final(self, train_data, cv_data, degree, alpha, iterations):
        X = train_data[:, 1:]
        y = train_data[:, 0]
        X_cv = cv_data[:, 1:]
        y_cv = cv_data[:, 0]
        X_ = self.binomial(X, degree)
        X_ = self.normalize(X_)
        W_in = self.generate_weight(X_)
        b_in = 0
        W, b, J = self.run_grad_descent(X_, W_in, b_in, y, alpha, iterations)
        print(f"The final cost after {iterations} iterations for degree {degree} is: {J}")
        print(f"The R2_score on cv set for degree {degree} is: {self.R2_score(self.predict(self.normalize_test(self.binomial(X, degree), self.binomial(X_cv, degree)), W, b), y_cv)}")
        return W, b, J

class Logistic_Regression():

    # def multiclass_labels(y, k):
    #     # remeber k is the no. of class
    #     y_ = np.zeros([len(y), k])
    #     for i in range(len(y)):
    #         x = y[i]
    #         y_[i][x] = 1
    #     return y_

    # def softmax(z):
    #     m = len(z)
    #     k = len(z[0])
    #     a = np.exp(z)
    #     # b = np.sum(a, axis=1)
    #     # return = a/b
    #     for i in range(len(z)):
    #         b = a[i]
    #         x = np.sum(b)
    #         a[i] = a[i]/x
    #     return a

    # def score(X, W, b):
    #     # sbse pehle scale down
    #     X = X/255
    #     # remember X has the shape (m, n) while W has shape(k, n) and b has shape (k, 1)
    #     score = X.dot(W.T)    # matrix multiplication
    #     score += b    # adding biases to each row of score
    #     # score returns a matrix of shape (m, k)
    #     return score

    # def gradient(X, y, y_hat):
    #     # X is the features here (m, n)
    #     # while y and y_hat has the shape (m, k)
    #     delta = y_hat - y
    #     dj_dw = ((X.T).dot(delta)).T
    #     dj_dw = dj_dw/len(X)
    #     dj_db = np.sum(delta, axis=0)
    #     dj_db = dj_db/len(X)
    #     return dj_dw, dj_db

    # def descent(X, y, alpha, iterations):
    #     m = len(X)
    #     n = len(X[0])
    #     k = len(y[0])
    #     W_in = (np.ones((k, n)))/100 # while W has the shape (k, n) and b has (k, 1)
    #     b_in = (np.ones(k))/100
    #     for i in range(iterations):
    #         y_hat = softmax(score(X, W_in, b_in))
    #         dj_dw, dj_db = gradient(X, y, y_hat)
    #         W_in -= alpha*dj_dw
    #         b_in -= alpha*dj_db
    #         if ((i+1)%100==0) :
    #             print(f"Cost at {i+1}th iteration is : {cost(y_hat, y)}")
    #     return W_in, b_in

    # def cost(y_hat, y):
    #     # both y_hat and y has the shape (m, k)
    #     epsilon = 1e-6
    #     cost_ = 0
    #     cost_ = np.mean(-np.sum(np.log(y_hat + epsilon)*y, axis=1))
    #     return cost_

    # def predict(X, W, b):
    #     score_ = score(X, W, b)
    #     y_hat = softmax(score_)
    #     ans = np.zeros((len(X)))
    #     for i in range(len(X)):
    #         p = 0
    #         index = -1
    #         for k in range(10):
    #             if(y_hat[i][k]>p):
    #                 index = k
    #                 p = y_hat[i][k]
    #         ans[i] = index
    #     return ans

    # def accuracy(y_pred, y_true):
    #     correct = 0
    #     for i in range(len(y_true)):
    #         if(y_pred[i]==y_true[i]):
    #             correct +=1
    #     return correct/len(y_true)

    def binary_encode(self, y, k):
        y_ = np.zeros((len(y),1))
        for i in range(len(y)):
            if(y[i]==k):
                y_[i] = 1
        return y_

    def sigmoid(self, z):
        a = 1/(1+np.exp(-z))
        return a

    def decision_boundary(self, y):
        y_ = y - 0.5 + 1e-8
        y_ = np.sign(y_)
        y_ = np.maximum(0.0, y_)
        return y_

    # def decision_boundary(y):
    #     y_ = np.zeros(y.shape)
    #     for i in range(len(y)):
    #         if(y[i]>=0.5):
    #             y_[i] = 1
    #     return y_

    def cost(self, y_hat, y_true):
        epsilon = 1e-7
        cost_ = np.mean(-y_true*np.log(y_hat+epsilon)-(1-y_true)*(np.log(1-y_hat+epsilon)))
        return cost_

    def gradient(self, X, Y_hat, Y):
        dw = np.zeros((len(X[0]), 1))
        db = 0
        delta = Y_hat - Y
        dw = np.dot(X.T, delta)
        db = np.mean(delta)
        dw = dw/len(X)
        return dw, db

    def model(self, x_train, y_train, k, alpha, iterations):
        y_ = self.binary_encode(y_train, k)
        W = np.zeros((len(x_train[0]), 1))
        b = 0
        # J = []
        for i in range(iterations):
            z = self.sigmoid(x_train.dot(W) + b)
            z = np.reshape(z, (len(z), 1))
            dj_dw, dj_db = self.gradient(x_train, z, y_)
            W -= alpha*dj_dw
            b -= alpha*dj_db
            # if((i+1)%10==0):
            #     J.append(self.cost(self.predict(x_train, W, b), y_train))
        return W, b

    def all_model(self, x_train, y_train, num_classes, alpha, iterations):
        W = []
        b = []
        # J = []
        x_train = x_train/255
        for i in range(num_classes):
            W_k, b_k= self.model(x_train, y_train, i, alpha, iterations)
            W.append(W_k)
            b.append(b_k)
            # J.append(J_k)
            z = x_train.dot(W_k) + b_k
            a = self.sigmoid(z)
            Y_true = self.binary_encode(y_train, i)
            Y_pred = self.decision_boundary(a)
            print(f"The accuracy for detecting {i} is: {self.evaluate(Y_pred, Y_true)}/{len(Y_true)}")
        return W, b

    def evaluate(self, y_hat, y_true):
        correct = sum(int(x==y) for x, y in zip(y_hat, y_true))
        return correct

    # def predict(self, X, W, b):
    #     return self.decision_boundary(self.sigmoid(np.dot(X, W) + b))

    def predict_cv(self, x_cv, W, b):
        m = len(x_cv[0])
        x_cv = x_cv/255
        y_pred = [self.sigmoid((x_cv.dot(np.reshape(W_, (m, 1)))) + b_) for W_, b_ in zip(W, b)]
        y = np.array((y_pred))
        y = np.reshape(y, (10, len(x_cv)))
        y = y.T
        y_pred = np.argmax(y, axis=1)
        y__ = np.reshape(y_pred, (len(y_pred), 1))
        return y__

class KNN():

    # def distance(a, b):
    #     ans = np.sum((a-b)**2)
    #     return ans

    # def closest_neighbours(X, X_train, y_train, K):
    #     pred_labels = np.zeros(len(X))
    #     for i in range(len(X)):
    #         dist_ = np.zeros((len(X_train), 2))
    #         for j in range(len(X_train)):
    #             dist = distance(X[i], X_train[j])
    #             dist_[j] = [dist, j]
    #         dist_ = dist_[dist_[:,0].argsort()]
    #         b = np.zeros(K, dtype="int64")
    #         for k in range(K):
    #             b[k] = y_train[int(dist_[k][1])]
    #         x = max(np.bincount(b))
    #         a = np.zeros((10))
    #         ans = -1
    #         for k in range(K):
    #             p = int(dist_[k][1])
    #             q = int(y_train[p])
    #             a[q] += 1
    #             if(a[q]==x):
    #                 ans = q
    #                 break
    #         pred_labels[i] = ans
    #     return pred_labels

    # def accuracy(y_pred, y_true):
    #     correct = 0
    #     for i in range(len(y_true)):
    #         if(y_pred[i]==y_true[i]):
    #             correct +=1
    #     return correct/len(y_true)

    def nearest_distance(self, x_train, x_test):
        dist = np.zeros((len(x_test), len(x_train)))

        # formula for euclidean distance ||x1-x2|| = (x1^2 + x2^2 - 2*x1.dot(x2.T))^(0.5)

        dist = (-2*np.dot(x_test, x_train.T)) # has the shape (len(x_test), len(x_train))
        dist += np.reshape(np.sum(x_test**2, axis = 1), (len(x_test), 1))  # has the shape (len(x_test), 1)
        dist += np.reshape((np.sum(x_train**2, axis = 1)).T, (1, len(x_train))) # has the shape (1, len(x_train))
        # here I tried to vectorize asmspossible by expanding the terms of eulidean distance for a vector

        return dist

    def best_k(self, train_data, test_data, max_k):
        x_train = train_data[:, 1:]
        y_train = train_data[:, 0]
        x_test = test_data[:, 1:]
        y_test = test_data[:, 0]
        dist = self.nearest_distance(x_train, x_test)
        for i in range(1, max_k+1):
            index = np.argpartition(dist, i-1)
            # using argpartition instead of argsort
            y_ = np.zeros((len(x_test), i))
            y_ = y_train[index[:,:i]]
            y_pred = [np.bincount(y).argmax() for y in y_]
            print(f"for k = {i} {self.evaluate(y_pred, y_test)}/{len(y_test)}")
    
    def predict(self, train_data, x_test, k):
        x_train = train_data[:, 1:]
        y_train = train_data[:, 0]
        dist = self.nearest_distance(x_train, x_test)
        index = np.argpartition(dist, k-1)
        y_ = np.zeros((len(x_test),k))
        y_ = y_train[index[:, :k]]
        y_pred = [np.bincount(y).argmax() for y in y_]
        y_pred = np.reshape(y_pred, (len(y_pred), 1))
        return y_pred

    def evaluate(self, y_pred, y_true):
        summ = sum(int(x==y) for x, y in zip(y_pred, y_true))
        return summ

class Neural_Network():

        # def sigmoid(z):
        #     a = 1.0/(1.0 + np.exp(-z))
        #     return a

        # def sigmoid_prime(z):
        #     return sigmoid(z)*(1.0-sigmoid(z))

        # def multiclass_labels(y, k):
        #     y_ = np.zeros([len(y), k])
        #     for i in range(len(y)):
        #         x = y[i]
        #         y_[i][x]=1
        #     return y_

        # def initialize_parameters(sizes):
        #     biases = [np.random.randn(j, 1) for j in sizes[1:]]
        #     weights = [np.random.randn(j, k)/np.sqrt(k) for j, k in zip(sizes[1:], sizes[:-1])]
        #     return weights, biases

        # def feedforward(X, weights, biases):
        #     X = np.reshape(X, (784,1))
        #     a = X
        #     for b, w in zip(biases, weights):
        #         z = np.dot(w, a)+b
        #         a = sigmoid(z)
        #     return a

        # def back_propagate(X, weights, biases, Y):
        #     X = np.reshape(X, (784,1))
        #     Y = np.reshape(Y, (10,1))
        #     a = X
        #     a_his = [a]
        #     z_his = []
        #     for w, b in zip(weights, biases):
        #         z = np.dot(w, a)+b
        #         z_his.append(z)
        #         a = sigmoid(z)
        #         a_his.append(a)
        #     dC_db = [np.zeros(b.shape) for b in biases]
        #     dC_dw = [np.zeros(w.shape) for w in weights]
        #     error = (a-Y)*sigmoid_prime(z)
        #     dC_db[-1] = error
        #     dC_dw[-1] = np.dot(error, a_his[-2].T)
        #     layers = len(weights)+1
        #     for l in range(2, layers):
        #         z = z_his[-l]
        #         error = np.dot(weights[-l+1].T, error)*sigmoid_prime(z_his[-l])
        #         dC_db[-l] = error
        #         dC_dw[-l] = np.dot(error, a_his[-l-1].T)
        #     return (dC_dw, dC_db)

        # def update_mini_batch(mini_batch, weights, biases, alpha):
        #     dC_dw = [np.zeros(w.shape) for w in weights]
        #     dC_db = [np.zeros(b.shape) for b in biases]
        #     X = mini_batch[:, 1:]
        #     X = X/255
        #     Y = mini_batch[:, 0]
        #     Y = multiclass_labels(Y, 10)
        #     for x, y in zip(X, Y):
        #         tem_dC_dw, tem_dC_db = back_propagate(x, weights, biases, y)
        #         dC_dw = [p+q for p, q in zip(dC_dw, tem_dC_dw)]
        #         dC_db = [r+s for r, s in zip(dC_db, tem_dC_db)]
        #     weights = [w - (alpha*dw)/len(mini_batch) for w, dw in zip(weights, dC_dw)]
        #     biases  = [b - (alpha*db)/len(mini_batch) for b, db in zip(biases, dC_db)]
        #     return weights, biases

        # def mini_batch_descent(train_data, sizes, epochs, alpha, mini_batch_size,):
        #     weights, biases = initialize_parameters(sizes)
        #     m = len(train_data)
        #     for i in range(epochs):
        #         np.random.shuffle(train_data)
        #         mini_batches = [train_data[k:k+mini_batch_size, :] for k in range(0, m, mini_batch_size)]
        #         for mini_batch in mini_batches:
        #             weights, biases = update_mini_batch(mini_batch, weights, biases, alpha)
        #         print ("Epoch {0}: {1} / {2}".format(i, evaluate(weights, biases, train_data), len(train_data)))
        #     return weights, biases

        # def evaluate(weights, biases, test_data):
        #     X = test_data[:,1:]
        #     Y = test_data[:,0]
        #     X = X/255
        #     count = 0
        #     test_results = [(np.argmax(feedforward(x, weights, biases)), y) for (x, y) in zip(X, Y)]
        #     return sum(int(x==y) for x, y in test_results)

    def relu(self, z):
        a = np.maximum(0.0, z)
        return a

    def relu_prime(self, z):
        a = np.sign(z)
        a = np.maximum(0.0, a)
        return a

    def softmax(self, z):
        a = np.exp(z-np.max(z))
        s = np.sum(a, axis=1)
        # b = a/np.expand_dims(s, 1)
        b = a.T/s
        return b.T

    def softmax_prime(self, z):
        return self.softmax(z)*(1-self.softmax(z))

    def multiclass_labels(self, y, k):
        y_ = np.zeros([len(y), k])
        for i in range(len(y)):
            x = y[i]
            y_[i][x]=1
        return y_

    def initialize_parameters(self, sizes):
        biases = [np.ones((j, 1)) for j in sizes[1:]]
        weights = [np.random.randn(j, k)/np.sqrt(k) for j, k in zip(sizes[1:], sizes[:-1])]
        return weights, biases

    def feedforward(self, X, weights, biases):
        a = X
        for b, w in zip(biases[:-1], weights[:-1]):
            z = np.dot(a, w.T)+b.T
            a = self.relu(z)
        z = np.dot(a, weights[-1].T) + biases[-1].T
        a = self.softmax(z)
        return a

    def back_propagate(self, X, weights, biases, Y):
        a = X
        a_his = [a]
        z_his = []
        for w, b in zip(weights[:-1], biases[:-1]):
            z = np.dot(a, w.T)+b.T
            z_his.append(z)
            a = self.relu(z)
            a_his.append(a)
        z = np.dot(a, weights[-1].T)+biases[-1].T
        z_his.append(z)
        a = self.softmax(z)
        a_his.append(a)
        dC_db = [np.zeros(b.shape) for b in biases]
        dC_dw = [np.zeros(w.shape) for w in weights]
        error = (a-Y)
        dC_db[-1] = np.reshape(np.sum(error, axis=0), (len(error[0]),1))
        dC_dw[-1] = np.dot(error.T, a_his[-2])
        layers = len(weights)+1
        for l in range(2, layers):
            z = z_his[-l]
            error = (np.dot(error, weights[-l+1])*self.relu_prime(z_his[-l]))
            dC_db[-l] = np.reshape(np.sum(error, axis=0), (len(error[0]),1))
            dC_dw[-l] = np.dot(error.T, a_his[-l-1])
        return (dC_dw, dC_db)

        # def back_propagate(X, weights, biases, Y):
        #     X = np.reshape(X, (784,1))
        #     Y = np.reshape(Y, (10,1))
        #     a = X
        #     a_his = [a]
        #     z_his = []
        #     for w, b in zip(weights[:-1], biases[:-1]):
        #         z = np.dot(w, a)+b
        #         z_his.append(z)
        #         a = relu(z)
        #         a_his.append(a)
        #     z = np.dot(weights[-1], a)+biases[-1]
        #     z_his.append(z)
        #     a = softmax(z)
        #     a_his.append(a)
        #     dC_db = [np.zeros(b.shape) for b in biases]
        #     dC_dw = [np.zeros(w.shape) for w in weights]
        #     error = (a-Y)
        #     dC_db[-1] = error
        #     dC_dw[-1] = np.dot(error, a_his[-2].T)
        #     layers = len(weights)+1
        #     for l in range(2, layers):
        #         z = z_his[-l]
        #         error = np.dot(weights[-l+1].T, error)*relu_prime(z_his[-l])
        #         dC_db[-l] = error
        #         dC_dw[-l] = np.dot(error, a_his[-l-1].T)
        #     return (dC_dw, dC_db)

    def update_mini_batch(self, mini_batch, weights, biases, alpha):
        dC_dw = [np.zeros(w.shape) for w in weights]
        dC_db = [np.zeros(b.shape) for b in biases]
        X = mini_batch[:, 1:]
        X = X/255
        Y = mini_batch[:, 0]
        Y = self.multiclass_labels(Y, 10)
        dC_dw, dC_db = self.back_propagate(X, weights, biases, Y)
        weights = [w - (alpha*dw)/len(mini_batch) for w, dw in zip(weights, dC_dw)]
        biases  = [b - (alpha*db)/len(mini_batch) for b, db in zip(biases, dC_db)]
        return weights, biases

    def mini_batch_descent(self, train_data, cv_data, sizes, epochs, alpha, mini_batch_size):
        weights, biases = self.initialize_parameters(sizes)
        m = len(train_data)
        accuracy = []
        for i in range(epochs):
            np.random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size, :] for k in range(0, m, mini_batch_size)]
            for mini_batch in mini_batches:
                weights, biases = self.update_mini_batch(mini_batch, weights, biases, alpha)
            acc =  self.evaluate(weights, biases, cv_data)
            print (f"Epoch {i}: {acc} / { len(cv_data)}")
            accuracy.append(acc)
        return weights, biases, accuracy

    def evaluate(self, weights, biases, test_data):
        X = test_data[:,1:]
        Y = test_data[:,0]
        X = X/255
        count = 0
        test_results = np.argmax(self.feedforward(X, weights, biases), axis=1)
        return np.sum(test_results==Y)

    def predict(self, X, weights, biases):
        norm_X = X/255
        a = self.feedforward(norm_X, weights, biases)
        return np.argmax(a, axis=1)

    def cv_accuracy(self, y_pred, y_cv):
        return np.sum(y_pred == y_cv)
    
class K_Means():

    # def index_closest_centroid(X, centroids):
    #     k = len(centroids)
    #     m, n = X.shape
    #     index_closest_centroid = np.zeros(m)
    #     for i in range(m):
    #         dis = np.zeros(k)
    #         for z in range(k):
    #             dis[z] = np.sum((X[i]-centroids[z])**2)
    #         index_closest_centroid[i] = int(np.argmin(dis))
    #     return index_closest_centroid

    # def mean_centroid(X, centroids):
    #     k = len(centroids)
    #     m = len(X)
    #     index_X = index_closest_centroid(X, centroids)
    #     average_centroid = np.zeros(centroids.shape)
    #     for z in range(k):
    #         count = 0
    #         for i in range(m):
    #             if (index_X[i] == z):
    #                 average_centroid[z] += X[i]
    #                 count += 1
    #         average_centroid[z] = (average_centroid[z])/count
    #     return average_centroid

    # def RunKMeans(X, in_centroids, iterations):
    #     K = len(in_centroids)
    #     centroids = in_centroids
    #     for i in range(iterations):
    #         centroids = mean_centroid(X, centroids)
    #     final_index =  index_closest_centroid(X, centroids)
    #     return centroids, final_index

    # def finalcost(X, centroids, index):
    #     m = len(X)
    #     cost_ = 0
    #     for i in range(m):
    #         cost_ += np.sum((X[i]-centroids[int(index[i])])**2)
    #     return cost_

    # def optimized_KMean(X, K, iterations, over_iter):
    #     cost_ = []
    #     centroids_ = []
    #     for i in range(over_iter):
    #         randomint = np.random.permutation(X.shape[0])
    #         centroids = X[randomint[:K]]
    #         centroids, index = RunKMeans(X, centroids, iterations)
    #         centroids_.append(centroids)
    #         cost = finalcost(X, centroids, index)
    #         cost_.append(cost)
    #     x = np.argmin(cost_)
    #     centroid = centroids_[x]
    #     return centroid, centroids_, cost_

    def cost(self, X, centroids):
        binary_indices = self.index_nearest_distance(X, centroids)
        centrd = np.dot(binary_indices, centroids)
        # print(X.shape)
        cost = np.mean(np.sum((X-centrd)**2, axis=1))
        return cost

    # def index_nearest_distance(X, centroids):
    #     dist = np.zeros((len(X), len(centroids)))
    #     # formula for euclidean distance ||x1-x2|| = (x1^2 + x2^2 - 2*x1.dot(x2.T))^(0.5)
    #     # print(centroids.shape)
    #     dist = (-2*np.dot(X, centroids.T)) # has the shape (len(X), len(centroids))
    #     dist += np.reshape(np.sum(X**2, axis = 1), (len(X), 1))  # has the shape (len(X), 1)
    #     dist += np.reshape((np.sum(centroids**2, axis = 1)).T, (1, len(centroids))) # has the shape (1, len(centroids))
    #     # here I tried to vectorize asmspossible by expanding the terms of eulidean distance for a vector
    #     # print(dist.shape)
    #     indices = np.argmin(dist, axis=1)
    #     return indices

    def index_nearest_distance(self, X, centroids):
        dist = np.zeros((len(X), len(centroids)))
        # formula for euclidean distance ||x1-x2|| = (x1^2 + x2^2 - 2*x1.dot(x2.T))^(0.5)
        # print(centroids.shape)
        dist = (-2*np.dot(X, centroids.T)) # has the shape (len(X), len(centroids))
        dist += np.reshape(np.sum(X**2, axis = 1), (len(X), 1))  # has the shape (len(X), 1)
        dist += np.reshape((np.sum(centroids**2, axis = 1)).T, (1, len(centroids))) # has the shape (1, len(centroids))
        # here I tried to vectorize asmspossible by expanding the terms of eulidean distance for a vector
        # this is not distance but its square
        # print(dist.shape)
        indices = np.argmin(dist, axis=1)
        binary_indices = np.zeros((len(X), len(centroids)))
        for i in range(len(X)):
            x = indices[i]
            binary_indices[i][x] = 1
        return binary_indices

    # def take_mean(X, centroids):
    #     indices = index_nearest_distance(X, centroids)
    #     # print(indices.shape)
    #     new_maps = np.concatenate((X, indices), axis=1)
    #     new_maps = new_maps[new_maps[:, -1].argsort()]
    #     new_X = new_maps[:,:-1]
    #     # print(new_X.shape)
    #     splt = [np.count_nonzero(indices==i) for i in range(len(centroids))]
    #     # print(spltt)
    #     for i in range(1, len(centroids)):
    #         splt[i] += splt[i-1]
    #     sub_arrays = np.split(new_X, splt)
    #     # new_sub_arrays = [np.nan_to_num(sub_arrays) for sub_array in sub_arrays]
    #     # print(sub_arrays[0].shape)
    #     # print(sub_arrays[1].shape)
    #     new_centroids = [np.mean(sub_array, axis=0) for sub_array in sub_arrays]
    #     new_centroids = np.nan_to_num(new_centroids)
    #     new_centroids = np.array(new_centroids)
    #     return new_centroids

    def take_mean(self, X, centroids):
        binary_indices = self.index_nearest_distance(X, centroids)
        # print(indices.shape)
        num_centroids = np.sum(binary_indices, axis=0)
        new_centroids = np.dot(binary_indices.T, X)
        new_centroids = np.divide(new_centroids.T, num_centroids.T)
        return new_centroids.T

    def run_k_means(self, X, k, iterations):
        initial_centroids = X[:k,:]
        # print(initial_centroids.shape)
        centroids = initial_centroids
        for i in range(iterations):
            centroids = self.take_mean(X, centroids)
        cost_ = self.cost(X, centroids)
        return centroids, cost_

    def normalize(self, X):
        mu     = np.mean(X, axis=0)
        sigma  = np.std(X, axis=0)
        X_norm = (X - mu) / sigma
        return X_norm

    def best_k(self, X, max_k, initializations, iterations):
        norm_X = self.normalize(X)
        cost_his = []
        centroids_his = []
        for i in range(1, max_k+1):
            tem_cost_his = np.zeros((initializations))
            tem_centroids_his = []
            np.random.shuffle(X)
            for j in range(initializations):
                ctrd, ct = self.run_k_means(norm_X, i, iterations)
                tem_centroids_his.append(ctrd)
                tem_cost_his[j] = ct
            index = np.argmin(tem_cost_his)
            cost_his.append(tem_cost_his[index])
            centroids_his.append(tem_centroids_his[index])
            print(f"Cost after {iterations} iterations for K = {i} is: {ct}")
        return centroids_his, cost_his
