# In[1]:
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
tr=0
ts=0
with open('train.txt','r') as train:
    while train.readline()!='':
        tr=tr+1
with open('test.txt','r') as test:
    while test.readline()!='':
        ts=ts+1
X_train = np.zeros((tr,256))
Y_train = np.zeros((tr,))
X_test = np.zeros((ts,256))
Y_test = np.zeros((ts,))
with open('train.txt','r') as train:
        for i in range(0,tr):
            line = train.readline().split()
            Y_train[i]=float(line[0])
            for j in range (0,256):
                X_train[i,j]=float(line[j+1])
with open('test.txt','r') as test:
        for i in range(0,ts):
            line = test.readline().split()
            Y_test[i]=float(line[0])
            for j in range (0,256):
                X_test[i,j]=float(line[j+1]) 
print(Y_train,Y_test)                

# In[2]:

image=np.reshape(X_train[130],(16,16))
plt.imshow(image)
# In[3]:
ar=[0]*10
axs = plt.subplots(10)
k=0    
for i in range(tr):
    if k==10:
        break
    index=int(Y_train[i])
    if ar[index]==0:
        image=np.reshape(X_train[i],(16,16))
        plt.subplot(3,4,k+1)
        plt.imshow(image)
        ar[index]+=1
        k+=1
plt.show()  

        
        
        
# In[4]:
total=0
#sum1=0
zeros=[]
pixel=[]
for i in range(tr):
    if int(Y_train[i])==0:
        zeros.append(X_train[i])
        array=np.reshape(X_train[i],(16,16))
        pixel.append(array[10][10])
        #sum1+=array[10][10]
        #sum1+=X_train[i][10*16 + 10]
        total+=1
#print(sum1/total)
mean_of_pixel_10_10 = stat.mean(pixel)
print(mean_of_pixel_10_10)
# In[5]:
var_of_pixel_10_10 = stat.variance(pixel)
print(var_of_pixel_10_10)
    
    
# In[6]:
rows, cols = (256, total) 
total_zeros_ar = [[0]*cols]*rows 
mean_of_zeros=[0]*256
variance_of_zeros=[0]*256
for j in range(256):
    for i in range(total):
        total_zeros_ar[j][i]=zeros[i][j]
    mean_of_zeros[j]=stat.mean(total_zeros_ar[j])
    variance_of_zeros[j]=stat.variance(total_zeros_ar[j])
print(mean_of_zeros[170],variance_of_zeros[170])
# In[7]:
image=np.reshape(mean_of_zeros,(16,16))
plt.imshow(image)

# In[8]:
image=np.reshape(variance_of_zeros,(16,16))
plt.imshow(image)
# In[9a]:
dif_by_num_ar = []
arr_divided_by_pixel_of_all_digits= []
how_many = [0]*10
arr_with_means = [] 
for i in range(10):
    arr_divided_by_pixel_of_all_digits.append([])
    dif_by_num_ar.append([])
    arr_with_means.append([])
    for j in range(256):
        arr_divided_by_pixel_of_all_digits[i].append([])
for i in range (tr):
    number = int(Y_train[i])
    how_many[number]+=1
    dif_by_num_ar[number].append(X_train[i])
for i in range(10):
    for j in range(256):
        for k in range(how_many[i]):
            arr_divided_by_pixel_of_all_digits[i][j].append(dif_by_num_ar[i][k][j])
for i in range(10):
    for j in range(256):
        arr_with_means[i].append(stat.mean(arr_divided_by_pixel_of_all_digits[i][j]))
for i in range(10):
    print("stats of digit ",i,":")
    for j in range(256):
        print("mean: ",i,j,arr_with_means[i][j],"variance: ",stat.variance(arr_divided_by_pixel_of_all_digits[i][j]))
#In[9b]:
k=0
for i in range(10):
    image=np.reshape(arr_with_means[i],(16,16))
    plt.subplot(3,4,k+1)
    plt.imshow(image)
    k+=1
plt.show()  

# In[10]:
smallest=1000000
smallest_index=0
distance = [0]*10
ar1=np.reshape(X_test[100],(16,16))
for i in range(10):
    ar2=np.reshape(arr_with_means[i],(16,16))
    distance[i]=np.linalg.norm(ar1-ar2)
    if smallest>distance[i]:
        smallest=distance[i]
        smallest_index=i
print(smallest_index,Y_test[100])
# In[11]:
suc=0
sm = [10000]*ts
sm_in = [0]*ts
rows, cols = (ts, 10) 
distance= [[0]*cols]*rows
ar1=[0]*ts
for i in range(ts):
    ar1[i]=np.reshape(X_test[i],(16,16))
    for j in range(10):
        ar2 = np.reshape(arr_with_means[j],(16,16))
        distance[i][j]=np.linalg.norm(ar1[i]-ar2)
        if sm[i]>distance[i][j]:
            sm[i]=distance[i][j]
            sm_in[i]=j
for i in range(ts):
    if sm_in[i]==int(Y_test[i]):
        suc+=1
    #print(sm_in[i],Y_test[i])
print(suc/ts)
# In[12]:
from sklearn.base import BaseEstimator, ClassifierMixin

class EuclideanClassifier(BaseEstimator, ClassifierMixin):  
    """Classify samples based on the distance from the mean feature value"""
    

    def __init__(self):
        self.X_mean_=[]

    def fit(self, X, y):
        in_of_256 = len(X[0])
        tr = len(y)
        dif_by_num_ar = []
        arr_divided_by_pixel_of_all_digits= []
        how_many = [0]*10  
        for i in range(10):
            arr_divided_by_pixel_of_all_digits.append([])
            dif_by_num_ar.append([])
            self.X_mean_.append([])
            for j in range(in_of_256):
                arr_divided_by_pixel_of_all_digits[i].append([])
        for i in range (tr):
            number = int(y[i])
            how_many[number]+=1
            dif_by_num_ar[number].append(X[i])
        for i in range(10):
            for j in range(in_of_256):
                for k in range(how_many[i]):
                    arr_divided_by_pixel_of_all_digits[i][j].append(dif_by_num_ar[i][k][j])
        for i in range(10):
            for j in range(in_of_256):
                self.X_mean_[i].append(stat.mean(arr_divided_by_pixel_of_all_digits[i][j]))
        return self
        raise NotImplementedError
        


    def predict(self, X):
        ts=len(X)
        suc=0
        sm = [10000]*ts
        sm_in = [0]*ts
        rows, cols = (ts, 10) 
        distance= [[0]*cols]*rows
        ar1=[0]*ts
        for i in range(ts):
            ar1[i]=X[i]
            for j in range(10):
                ar2 = self.X_mean_[j]
                distance[i][j]=np.linalg.norm(ar1[i]-ar2)
                if sm[i]>distance[i][j]:
                    sm[i]=distance[i][j]
                    sm_in[i]=j
        return sm_in
        raise NotImplementedError
    
    def score(self, X, y):
        hulio=self.predict(X)
        total=0
        pos=0
        for i in hulio:
            if i==y[pos]:
                total+=1
            pos+=1
        return (total/pos)*100
        raise NotImplementedError
        
        
        
r1 = EuclideanClassifier()
r1.fit(X_train,Y_train)
print(r1.score(X_test,Y_test))
# In[13a]:
parts = tr//5
r1 = EuclideanClassifier()
r1.fit(X_train[parts:],Y_train[parts:])
print(r1.score(X_train[:parts],Y_train[:parts]))
for i in range(2,6):
    r1= EuclideanClassifier()
    r1.fit(np.concatenate((X_train[:(i-1)*parts],X_train[i*parts:]), axis=0),np.concatenate((Y_train[:(i-1)*parts],Y_train[i*parts:]), axis=0)) 
    print(r1.score(X_train[(i-1)*parts:i*parts],Y_train[(i-1)*parts:i*parts]))


# In[13b]:
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = X_train

n_samples, n_features = data.shape
n_digits = len(np.unique(Y_train))
labels = Y_train


# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1


xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
# In[13c]:
from sklearn.model_selection import learning_curve
import warnings

    
def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 1)):
    plt.figure()
    plt.title("Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


clf = EuclideanClassifier()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    train_sizes, train_scores, test_scores = learning_curve(clf, X_train, Y_train, cv = 10, n_jobs = 4,train_sizes=np.linspace(.1, 1.0, 5))    


plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0, 100))





