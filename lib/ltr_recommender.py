#!/usr/bin/env python
"""
Module that provides the main functionalities of LTR approach to CBF.
"""
import time
import numpy
from sklearn import svm
import numpy.ma as ma
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

numpy.set_printoptions(threshold=numpy.inf)
        

class LTRRecommender(object):
    """
      A class that takes in the Labeled rating matrix and theta(the output of LDA)
    """
    def __init__(self,n_users,n_docs,theta,peerpaper_search_strategy,sparse_documents_similarity, ratings, n_peers):
      """
      """
      self.n_users = n_users
      self.n_docs = n_docs
      self.user_model = [None for user in range(n_users)]
      self.theta = theta
      self.peerpaper_search_strategy = peerpaper_search_strategy
      self.sparse_documents_similarity = sparse_documents_similarity
      self.test_indices = [None for user in range(n_users)]
      self.ratings = ratings
      self.n_peers = n_peers
      #using random neg paper indices to experiment in predict
      #self.random_neg_indices= [None for user in range(n_users)]

    def train(self, test_mask):
      """
      Train the LTR Recommender.
      """
      labeled_r = None
      train_data = self.ratings.copy()
      #zero values are also considered as a mask
      train_data[train_data == 0] = 2
      train_data = ma.masked_array(train_data , test_mask)
      for user in range(self.n_users):
        user_rated_indices = numpy.where(train_data[user] == 1)[0]
        non_rated_indices = numpy.where(train_data[user] == 2)[0]
        peers_per_paper = self.n_peers
        user_rated_peer_matrix = None
        if (self.peerpaper_search_strategy == "paper_based"):
          user_rated_peer_matrix =  self.paper_based_peerpapers(user_rated_indices,peers_per_paper)
        elif (self.peerpaper_search_strategy == "random"):
          user_rated_peer_matrix = self.random_peerpapers(user_rated_indices,non_rated_indices, peers_per_paper)

        #builds a pair matrix for every user
        pair_matrix,labels = self.build_pairs_per_user(user_rated_peer_matrix)
        #fit Model that is SVM
        self.user_model[user] = svm.SVC(kernel="linear", probability = True)
        self.user_model[user].fit(pair_matrix, labels)

    def build_pairs_per_user(self, user_rated_peer_matrix):
      """
      :param user_rated_peer_matrix: (user_rated_indices.size ,peers_per_paper + 1)
      :return: pair_matrix,labels
      """
      user_rated_indices = user_rated_peer_matrix[:,0]
      #adding a new dimesion to vectorize
      peer_paper_indices = user_rated_peer_matrix[:,1:]

      rated_docs_theta = self.theta[user_rated_indices]
      rated_docs_theta = rated_docs_theta[:,numpy.newaxis,:]
      peer_paper_theta = self.theta[peer_paper_indices]

      pair_matrix_positive = rated_docs_theta - peer_paper_theta
      pair_matrix_positive = numpy.reshape(pair_matrix_positive , (pair_matrix_positive.shape[0] * pair_matrix_positive.shape[1],pair_matrix_positive.shape[2]))
      labels_ones =  numpy.ones(pair_matrix_positive.shape[0])

      #TODO: possible optimization here

      pair_matrix_negative = peer_paper_theta - rated_docs_theta
      pair_matrix_negative = numpy.reshape(pair_matrix_negative,(pair_matrix_negative.shape[0] * pair_matrix_negative.shape[1], pair_matrix_negative.shape[2]))
      labels_zeros = numpy.zeros(pair_matrix_negative.shape[0])

      pair_matrix = numpy.concatenate((pair_matrix_positive,pair_matrix_negative), axis =0 )
      labels = numpy.append(labels_ones,labels_zeros)
      return pair_matrix, labels

    def paper_based_peerpapers(self,user_rated_indices,peers_per_paper):
      """
      :param user_rated_indices:(1d array) user rated document indices
      :param peers_per_paper:(int) hyper-parameter to search for peers per user
      :param non_rated_indices: (1d array) user non rated document indices
      :return:
       user_rated_peer_matrix : (user_rated_indices.size,peers_per_paper + 1)
      """

      user_rated_peer_matrix = numpy.zeros((user_rated_indices.size, peers_per_paper + 1))
      user_rated_peer_matrix[:, 0] = user_rated_indices.copy()
      user_sorted_dense_document_similarity = numpy.argsort(self.sparse_documents_similarity[user_rated_indices].toarray())
      user_rated_peer_matrix[:, 1:] = user_sorted_dense_document_similarity[:, :peers_per_paper]
      user_rated_peer_matrix = user_rated_peer_matrix.astype(int)

      return user_rated_peer_matrix

    def random_peerpapers(self,user_rated_indices,non_rated_indices,peers_per_paper):
      """
      :param user_rated_indices: (1d array) user rated document indices
      :param peers_per_paper:(int) hyper-parameter to search for peers per user
      :param non_rated_indices: (1d array) user non rated document indices
      :return:
      user_rated_peer_matrix : (user_rated_indices.size,peers_per_paper + 1)
      """

      user_rated_peer_matrix = numpy.zeros((user_rated_indices.size, peers_per_paper + 1))
      user_rated_peer_matrix[:, 0] = user_rated_indices.copy()
      #todo:vectorize below code
      for x in range(user_rated_indices.size):
        user_rated_peer_matrix[x, 1:] = numpy.random.choice(non_rated_indices,peers_per_paper,replace=False)
      user_rated_peer_matrix = user_rated_peer_matrix.astype(int)

      return user_rated_peer_matrix

    def predict(self,test_mask):
        """
        Predict ratings for every user and item.

        :returns: A (user, document) matrix of predictions
        :rtype: ndarray
        """

        predictions = numpy.zeros((self.n_users,self.n_docs))
        prediction_scores = numpy.zeros((self.n_users,self.n_docs))
        for user in range(self.n_users):
          #half vectorized version (still outer loop there) :(
          #docs_to_test = self.theta[ self.test_indices[user] ]
          test_indices_user = numpy.where(test_mask[user])[0]
          docs_to_test = self.theta[test_indices_user ]
          predictions[user , test_indices_user ] = self.user_model[user].predict(docs_to_test)
          prediction_scores[user,test_indices_user  ] = self.user_model[user].predict_proba(docs_to_test)[:, 1]

        return predictions,prediction_scores


""" 
          
      if (self.peerpaper_search_strategy == "random"):
        labeled_r = self.put_random_negatives(train_data)
      elif (self.peerpaper_search_strategy == "pairwise"):
        labeled_r = self.put_pairwise_negatives(train_data)
      

    Fallback Solution
    def put_random_negatives(self,rating_matrix):

      input:
          rating matrix
      :returns:
          randomly put negative ratings and return the rating_matrix
      :rtype: int[][]

      for user in range(self.n_users):
        zero_indices = numpy.where(rating_matrix[user] == 2 )[0]
        num_positive_ids = numpy.count_nonzero(rating_matrix[user] == 1)
        random_negatives = numpy.random.choice(zero_indices,num_positive_ids,replace=False)
        for random_id in random_negatives:
          rating_matrix[user,random_id] = -1          
      return rating_matrix

    def put_pairwise_negatives(self,rating_matrix):

      input:
          rating matrix
      :returns:
          randomly put negative ratings and return the rating_matrix
      :rtype: int[][]

      er_model = [None for user in range(n_users)]
      for user in range(self.n_users):
        ones_indices = numpy.where(rating_matrix[user] == 1)[0]
        numpy.random.shuffle(ones_indices)
        num_positive_ids = numpy.count_nonzero(rating_matrix[user] == 1)
        neg_ids = []
        for indice in ones_indices:
          for document_id in self.sorted_documents_similarity[indice]:
            # and not ma.is_masked(rating_matrix[user,document_id]):
            if (document_id not in neg_ids) and rating_matrix[user,document_id] == 2 :
              neg_ids.append(document_id)
              break
        for neg_id in neg_ids:
          rating_matrix[user,neg_id] = -1          
      return rating_matrix
"""
""" 
def build_pairs(self,labeled_r,theta ,user_id):
    This function builds the pairs from the labeled rating matrix such that each pair is a positive and negative entry
    :param matrix labeled_r : labeled rating matrix consisting of 1,-1 and 0
    :param theta : document distribution matrix
    :returns : pair matrix of all positive and negative ratings and labels

  pos_ids = numpy.where( labeled_r[user_id,:] == 1)[0]
  neg_ids = numpy.where( labeled_r[user_id,:] == -1)[0]
  #shuffling them to destroy any order
  numpy.random.shuffle(pos_ids)
  numpy.random.shuffle(neg_ids)

  #self.random_neg_indices[user_id] = numpy.random.choice(neg_ids)

  pair_matrix = numpy.zeros(( len(pos_ids) * len(neg_ids) * 2 , theta.shape[1] ))
  labels_ones = numpy.ones( len(pos_ids) * len(neg_ids) )
  labels_zeros = numpy.zeros(len(pos_ids) * len(neg_ids) )
  labels = numpy.append(labels_ones,labels_zeros)
  counter = 0
  for pos_id in pos_ids:
    for neg_id in neg_ids:
      pair_matrix[counter] = theta[pos_id] - theta[neg_id]
      counter = counter + 1
  for pos_id in pos_ids:
    for neg_id in neg_ids:
      pair_matrix[counter] = theta[neg_id] - theta[pos_id]
      counter = counter + 1
  return pair_matrix,labels
"""

