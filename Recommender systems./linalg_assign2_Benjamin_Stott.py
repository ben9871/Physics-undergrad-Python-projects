import numpy
import pandas as pd
import matplotlib.pyplot as plt # data visualization
import seaborn as sns           # statistical data visualization
import scipy.sparse as sp
from scipy.sparse.linalg import svds

#import relevant libraries


header_list = ["user_id", "age", "gender", "occupation", "zipcode"]
user = pd.read_table('u.user', skiprows=None,header=None, sep='|', lineterminator='\n', names=header_list)

user = pd.DataFrame(user)
print(user)

header_list = ["movie id",  "movie title",  "release date",  "video release date",
              "IMDb URL",  "unknown",  "Action",  "Adventure",  "Animation",
              "Children's",  "Comedy" , "Crime" , "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance" ,"Sci-Fi" , "Thriller" ,"War", "Western" ]
item = pd.read_table('u.item', skiprows=None,header=None, sep='|', lineterminator='\n', names=header_list)

item = pd.DataFrame(item)
print(item)

header_list = ["user_id", "item_id", "rating", "timestamp"]
test = pd.read_table('u.data', skiprows=None,header=None, delim_whitespace=True, names=header_list)


test = pd.DataFrame(test)
print(test)

#import data from different sets as dataframes, giving columns titles
movie_matrix = test.pivot_table(index='user_id', columns='item_id', values='rating')
movie_matrix.head()
print(movie_matrix)

#the ratings matrix R can be created with users on the rows and movies on columns into a pivot table.
#it is now fit to be passed to Chen's routine

#plt.figure(1)

steps_total = []
error_rmse =[]
error_rmse_rand=[]
#empty lists for graphing purposes


def matrix_factorization(R, P, Q, K, steps=350, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''
    Q = Q.T
    st=0
     
    error=0
    #steps_total =[]
    #error_rmse = []
    for step in range(steps):
       
        st+=1
        steps_total.append(st)
        #plt.figure(1)
        #plt.plot( st, error)
        sq_err = []
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    #sq_err = []
                    # calculate error
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    #error += (pow(R[i][j]  - numpy.dot(P[i,:],Q[:,j], 2)))
                    error_new = eij**2 #numpy.sqrt(eij**2)
                    #error_rmse.append(error_new) 
                    sq_err.append(error_new)           

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = numpy.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
                        
        # 0.001: local minimum
        if e < 0.001:

            break
        mean_sqr = numpy.mean(numpy.array(sq_err))
        rms = numpy.sqrt(mean_sqr)
        
        error_rmse_rand.append(rms)
        #error_rmse= numpy.array(error_rmse)
        #error_rmse_rand = numpy.sqrt(numpy.mean(pow(error_rmse,2)))
        #plt.plot(st, error_new, marker = "o", markersize =10)
        #plt.show()

    return P, Q.T
#chen's routing with a very slight alteration, rmse is extracted for each step along with step index for graphing purposes


steps_total_rand = steps_total



R = numpy.array(movie_matrix)
print(R)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])
# Num of Features
K = 4

 
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)
#define initial random matrices p and q for user tastes and movies with 4 k factors
 
nP, nQ = matrix_factorization(R, P, Q, K)
print(nP)
nR = numpy.dot(nP, nQ.T)
print(nR)
nR = pd.DataFrame(nR)
#reconstructing the original matrix
print(movie_matrix)
#plt.plot(steps, error_rmse)

#nR.to_csv("~/Desktop/matrix/ml-100k/predictedset.csv", sep='\t')
#movie_matrix.to_csv("~/Desktop/matrix/ml-100k/original.csv", sep='\t')
#can be saved to csv if one so chooses since this is a lengthy process using the full dataset and don't want to repeat each time

plt.figure(1)
plt.scatter(steps_total_rand, error_rmse_rand, marker = "o",edgecolor="grey", linewidth=0.1, label="random")
plt.title("Rmse vs iteration number for random initial P, Q vector guess")
plt.xlabel("iteration number")
plt.ylabel("RMSE")
plt.legend(loc="best")
plt.show()

#a plot of steps vs error for the random initial vectors P and Q



simil=[]
for i in range(len(nQ)):
    a =  numpy.dot(nQ[0],nQ[i])/((numpy.linalg.norm(nQ[0]))*(numpy.linalg.norm(nQ[i])))
    simil.append(a)

#for a movie in Q, here movie[0] chosen on a whim, we take the cosine similarity with the other rows and form a list from them


simil = numpy.array(simil)
print(simil)
def largest_indices(ary, n):
   
    flat = ary.flatten()
    indices = numpy.argpartition(flat, -n)[-n:]
    indices = indices[numpy.argsort(-flat[indices])]
    return numpy.unravel_index(indices,ary.shape)

#20 largest values are extracted from this list using a largest indices function

most_movies = largest_indices(simil,20)
most_movies = most_movies[0]
for i in most_movies:
    print(i)


print(most_movies)
#test to see this works

fac1=[]
fac2=[]
#empty lists for factor 1 and 2 of 20 similar movies
print(nQ[1][0])
print(nQ)
for i in most_movies:
    k1= nQ[i][0]
    k2= nQ[i][1]
    fac1.append(k1)
    fac2.append(k2)
#factor 1 and 2 made into separate lists for each movie



header_list = ["user_id", "item_id", "rating", "timestamp"]
test1 = pd.read_table('u1.test', skiprows=None,header=None, delim_whitespace=True, names=header_list)


test1 = pd.DataFrame(test1)

test_1k = list(numpy.array(test1["item_id"]))
import random
test_1k = [x for x in test_1k if x not in most_movies]
test_indices_1k = random.sample(test_1k, 1000)

#u1.test data imported via pandas, from which 1000 movie indices are extracted randomly to be plotted against the 20 similar movies
#this will allow us to observe the difference between similar data on a factor graph and dissimilar data
#should observe a large spread and almost random placement of dissimilar movies while similar ones will be in a straight vector of maximum variance

fac1_t=[]
fac2_t=[]
for i in test_indices_1k:
    k1= nQ[i][0]
    k2= nQ[i][1]
    fac1_t.append(k1)
    fac2_t.append(k2)
#factor 1 and factor 2 for the 1000 random movies are extracted


plt.figure(2)
plt.scatter(fac1, fac2, label="similar movies")
most_movies3=[]
for i in most_movies:
    a= i +1
    most_movies3.append(a)
#pythonic indexing was used for the movies(which is not the case in the datafile) so at the end the real values n+1 are returned



most_movies= most_movies3

for i, txt in enumerate(most_movies):
    plt.annotate(txt, (fac1[i], fac2[i]))

#movie number is plotted for each of the 20 similar movies
    
plt.scatter(fac1_t, fac2_t, label = "random movies")    
plt.xlabel("Factor 1 (k1)")
plt.ylabel("Factor 2 (k2)")
plt.title("factor 1 vs factor 2 for 1000 random movies and 20 similar movies")
plt.legend(loc="best")

#the plot of these factors is then created
plt.show()

#an attempt at implementing the svd initial guess is found below
#part (7) was done prior to this since both random initial guess and svd return optimal p and q vectors given the right amount of iterations

steps_total = []
error_rmse =[]
error_rmse_rand=[]
#relevant lists are emptied to prep for graphing


movie_matrix = movie_matrix.replace(numpy.nan, 0)
#SVD can only work with a full matrix, as such all the nan values will be replaced by zeros


R2 = numpy.array(movie_matrix)

Q1, sigma, Q2T = numpy.linalg.svd(R2)
#the U s and Vt vectors


print("\nOriginal Matrix\n")
#print(a)
print("\nQ1\n")
print(Q1)
print("\nSigma\n")


S  = numpy.zeros(shape=(len(R2),len(R2[0])))
for i in range(len(R2)):
    S[i,i] = sigma[i]
print(S)
#S must be diagonalised before use


p= numpy.dot(Q1,S)
# we can take p to be the dot product of Q1.S and Q to be Q2T for the initial guess
#from then on Chen's routine remains untouched and is implemented similarly to before

q= Q2T
nP, nQ = matrix_factorization(R2, p, q, K)

nR = numpy.dot(nP, nQ.T)
print(nR)
nR = pd.DataFrame(nR)
#print(movie_matrix)
#plt.plot(steps, error_rmse)

#nR.to_csv("~/Desktop/matrix/ml-100k/predictedset.csv", sep='\t')
#once again data can be extracted in a separate file if one so wishes
plt.figure(2)
plt.scatter(steps_total, error_rmse_rand, marker = "o", edgecolor="grey", linewidth=0.1, label ="SVD")
plt.title("Rmse vs iteration number for SVD initial P, Q vector guess")
plt.xlabel("iteration number")
plt.ylabel("RMSE")
plt.legend(loc="best")
#the plot of rmse vs steps is made for the case of the svd initial guess
plt.show()


