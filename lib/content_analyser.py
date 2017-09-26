"""
A module that provides functionality for analysing document information
"""
import numpy
import math
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class Content_Analyser(object):
    """
    A class that analyse the content of documents
    """

    @staticmethod
    def get_document_distribution(term_freq ,lda_topics):
      """
        This function calculates the document distribution matrix and returns it
      """
      lda = LatentDirichletAllocation(n_topics=lda_topics, max_iter=5,
                                      learning_method='online',
                                      learning_offset=50., random_state=0,
                                      verbose=0)
      document_distribution = lda.fit_transform(term_freq)
      return  document_distribution

    @staticmethod
    def get_sorted_cosine_sim(theta):
      """
        This function calcualte the cosine similarity between the theta matrix and return them in sorted manner
      """
      theta_sparse = csr_matrix(theta)
      sparse_cosine_documents_similarity = cosine_similarity(theta_sparse,dense_output = False)
      #deprecated#sorted_sim_indices = numpy.argsort(pairwise_cosine_sim)
      return sparse_cosine_documents_similarity