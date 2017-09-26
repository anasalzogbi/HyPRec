#!/usr/bin/env python3
"""
A module to run different recommenders.
"""
import sys
import itertools
import numpy
import time
from optparse import OptionParser
#from lib.evaluator import Evaluator
from lib.evaluator_ltr import LTR_Evaluator
#from lib.grid_search import GridSearch
#from lib.recommender_system import RecommenderSystem
#from util.abstracts_preprocessor import AbstractsPreprocessor
from util.data_parser import DataParser
#from util.recommender_configuer import RecommenderConfiguration
#from util.model_initializer import ModelInitializer
#from util.runs_loader import RunsLoader
from scipy import sparse


class RunnableRecommenders(object):
    """
    A class that is used to run recommenders.
    """
    def __init__(self, use_database=True, verbose=True, load_matrices=True, dump=True, train_more=True,
                 random_seed=False,lda_topics=50 , n_peers=2, config=None):
        """
        Setup the data and configuration for the recommenders.
        """
        dataParser = DataParser(10)
        if use_database:           
            self.ratings = numpy.array(dataParser.get_ratings_matrix())
            word_to_count, article_to_word,article_to_word_to_count = dataParser.get_word_distribution()
        else:
            abstracts = {0: 'hell world berlin dna evolution', 1: 'freiburg is green',
                         2: 'the best dna is the dna of dinasours', 3: 'truth is absolute',
                         4: 'berlin is not that green', 5: 'truth manifests itself',
                         6: 'plato said truth is beautiful', 7: 'freiburg has dna'}

            vocab = set(itertools.chain(*list(map(lambda ab: ab.split(' '), abstracts.values()))))
            w2i = dict(zip(vocab, range(len(vocab))))
            word_to_count = [(w2i[word], sum(abstract.split(' ').count(word)
                                             for doc_id, abstract in abstracts.items())) for word in vocab]
            article_to_word = list(set([(doc_id, w2i[word])
                                        for doc_id, abstract in abstracts.items() for word in abstract.split(' ')]))
            article_to_word_to_count = list(set([(doc_id, w2i[word], abstract.count(word))
                                                 for doc_id, abstract in abstracts.items()
                                                 for word in abstract.split(' ')]))
            self.documents, self.users = 8, 10
            self.n_items =  self.documents
            self.ratings = numpy.array([[int(not bool((article + user) % 3))
                                         for article in range(self.documents)]
                                        for user in range(self.users)])
        
        self.verbose = verbose
        self.load_matrices = load_matrices
        self.dump = dump
        self.train_more = train_more
        self.random_seed = random_seed
        self.lda_topics = lda_topics
        self.n_peers = n_peers
        self.n_users,self.n_docs = self.ratings.shape
        #initialising the ltr evaluator
        self.evaluator_ltr = LTR_Evaluator(self.n_users,self.ratings)

        articles, words, counts = zip(*article_to_word_to_count)
        num_items = self.n_docs
        num_vocab = max(map(lambda inp: inp[0], word_to_count)) + 1
        
        self.term_freq = sparse.coo_matrix((counts, (articles, words)),shape=(num_items , num_vocab) ).tocsr()


    def run_ltr_recommender(self):
        from lib.ltr_recommender import LTRRecommender 
        from lib.content_analyser import Content_Analyser 
        
        """
        Runs LTR Recommender
        """
        n_folds = 5
        #peerpaper_search_strategy = "random"
        peerpaper_search_strategy = "paper_based"

        #k_fold_test_mask: boolean mask of shape (n_folds,n_users,n_docs)
        k_fold_test_mask = self.evaluator_ltr.generate_k_fold_test_mask(self.ratings,n_folds)

        theta = Content_Analyser.get_document_distribution(self.term_freq,self.lda_topics)
        #sorted_documents_similarity: matrix of document to document similarity ofshape(n_docs,n_docs)
        sparse_documents_similarity = None

        if(peerpaper_search_strategy == "paper_based"):
            sparse_documents_similarity = Content_Analyser.get_sorted_cosine_sim(theta)

        for fold in range(n_folds):
          ltr_recommender = LTRRecommender(self.n_users,self.n_docs,theta,peerpaper_search_strategy,sparse_documents_similarity ,self.ratings,self.n_peers)
          ltr_recommender.train(k_fold_test_mask[fold])
          predictions, prediction_scores = ltr_recommender.predict(k_fold_test_mask[fold])
          mrr_at_five = self.evaluator_ltr.calculate_mrr(5,predictions , prediction_scores , k_fold_test_mask[fold])
          ndcg_at_five = self.evaluator_ltr.calculate_ndcg(5,predictions)
          report_str = "Report : mrr@5 {:.5f} , ndcg@5 {:.5f} "
          print(report_str.format(mrr_at_five,ndcg_at_five))
          predictions = None
          prediction_scores = None
          ltr_recommender = None


if __name__ == '__main__':
    parser = OptionParser("runnables.py [options] [recommenders]\n\nRecommenders:\n\trecommender\n\tcollaborative"
                          "\n\tgrid_search\n\tlda\n\tlda2vec\n\tsdae\n\texperiment\n\texperiment_with_gridsearch")
    parser.add_option("-d", "--use-database", dest="db", action='store_true',
                      help="use database to run the recommender", metavar="DB")
    parser.add_option("-a", "--all", dest="all", action='store_true',
                      help="run every method", metavar="ALL")
    parser.add_option("-s", "--save", dest="dump", action='store_true',
                      help="dump the saved data into files in matrices/", metavar="DUMP")
    parser.add_option("-l", "--load", dest="load", action='store_true',
                      help="load saved models from files in matrices/", metavar="LOAD")
    parser.add_option("-v", "--verbose", dest="verbose", action='store_true',
                      help="print update statements during computations", metavar="VERBOSE")
    parser.add_option("-r", "--random_seed", dest="random_seed", action='store_true',
                      help="Set the seed to the current timestamp if true.", metavar="RANDOMSEED")
    parser.add_option("-t", "--lda_topics", dest="lda_topics", action='store', type="int",
                      help="Set the number of lda topics", metavar="LDATOPICS")
    parser.add_option("-p", "--peers", dest="n_peers", action='store', type="int",
                      help="Set the number of peer papers", metavar="PEERS")

    options, args = parser.parse_args()
    use_database = options.db is not None
    use_all = options.all is not None
    load_matrices = options.load is not None
    verbose = options.verbose is not None
    dump = options.dump is not None
    train_more = True
    random_seed = options.random_seed is not None
    lda_topics = 50
    n_peers = 2
    if random_seed is True:
        numpy.random.seed(int(time.time()))
    if options.lda_topics is not None:
        lda_topics = options.lda_topics
    if options.n_peers is not None:
      n_peers = options.n_peers
    runnable = RunnableRecommenders(use_database, verbose, load_matrices, dump, train_more, random_seed , lda_topics , n_peers)
    if use_all is True:
        runnable.run_ltr_recommender()
        sys.exit(0)
    found_runnable = False
    for arg in args:
        if arg == 'ltr':
            runnable.run_ltr_recommender()
            found_runnable = True     
        else:
            print("'%s' option is not valid, please use one of "
                  "['recommender', 'collaborative', 'grid_search', 'lda', 'lda2vec', 'experiment', "
                  "'sdae', 'experiment_with_gridsearch']" % arg)
    if found_runnable is False:
        runnable.run_ltr_recommender()
