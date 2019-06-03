'''
Classifying documents using a naive version of Bayes' theorem
Answering the question: "Given this word (in the document), what is 
the probability that it belongs in a certain category?"

Bayes theorem:
P(A | B) = (P(B | A) * P(A)) / P(B)


For this class and the functions within it, our goal is to build the 
pieces to arrive at the following function for P(A|B):

P(Category | Word) = P(Word | Category) * P(Category) / P(Word)

Where we assume away the denominator, P(Word), because this is a naive
version of Bayes's theorem that assumes each variable is independent
and thus the denominator is the same across all of our categories.
In other words we ignore the denominator because we're comparing 
relative probabilities across the numerator of our function.

We define P(B | A) as:

P(Word | Category) = (Word Frequency in the Document + 1) / (Word Frequency 
    Across All Documents in the Category + Number of Words Across All Docs)

Using Laplacian smoothing, we add 1 to the numerator above to make sure none are
ever 0, and add the number of words across all reviews to make sure we never
divide by 0

'''
import numpy as np
import pandas as pd
import time

stop_words = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again',
              'there', 'about', 'once', 'during', 'out', 'very', 'having',
              'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
              'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off',
              'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
              'the', 'themselves', 'until', 'below', 'are', 'we', 'these',
              'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her',
              'more', 'himself', 'this', 'down', 'should', 'our', 'their',
              'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she',
              'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same',
              'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
              'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
              'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
              'just', 'where', 'too', 'only', 'myself', 'which', 'those',
              'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs',
              'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further',
              'was', 'here', 'than'}

class DocumentClassification():
    def __init__(self, data, word_loc, category_loc, stopwords=False):
        
        # Expects data is a pandas dataframe
        self.data = data

        # The location of the words per document
        # Expects the words are found as a column in the df
        self.word_loc = word_loc

        # The location of the target variable, category
        # Expects category is found as a column in the df
        self.category_loc = category_loc
        self.stopwords = stopwords

        self.p_categories = self.get_prob_categories()
        self.V = self.len_vocabulary()
        self.word_freqs = self.calc_word_freqs()

        


    # Calculating P(Category): probability that a document falls in each category
    # Returns a dictionary with categories as keys
    def get_prob_categories(self):

        p_category = dict(self.data[self.category_loc].value_counts(normalize = True))
        return p_category


    # Calculating the Word Frequency in the Document:
    # Returns a dictionary with words as keys, frequency as values
    # Input doc_content: expects the location of the document (tested for cells
    # in self.data containing content as a single line)
    def count_words(self, doc_content):
        count = {}
        
        # Adding 1 to the count for the word:
        # If the word is in the 'count' dictionary, grabs that value
        # If the word is not yet in the 'count' dictionary, sets value to 0
        # Then, adds 1 to the value
        for word in doc_content.split():
            count[word] = count.get(word, 0) + 1
            
        return count


    # Calculating the Word Frequency Across All Docs in that Category:
    # Returns a nested dictionary with categories as keys, where the values 
    # are another dictionary with each word within documents of that category
    # as the keys and the frequency of each word as values
    def calc_word_freqs(self):
        word_freq = {}

        categories = self.data[self.category_loc].unique()

        # Putting our cats in the bag (nested dictionary)
        for cat in categories:
            temp_df = self.data[self.data[self.category_loc] == cat]

            bag = {}
            for row in temp_df.index:
                doc = temp_df[self.word_loc][row]
                for word in doc.split():
                    bag[word] = bag.get(word, 0) + 1
            word_freq[cat] = bag

        if self.stopwords == True:
            list_keys = list(word_freq.keys())
            for key in list_keys:
                if key in stop_words:
                    del word_freq[key]
        
        return word_freq


    # Calculating the Number of Words Across All Docs for Laplacian smoothing,
    # by counting the total number of unique words in all documents
    # Returns the length of the vocabulary across all documents
    def len_vocabulary(self):
        
        # Making vocabulary a set so it only keeps unique elements/words
        vocabulary = set()

        for text in self.data[self.word_loc]:
            for word in text.split():
                vocabulary.add(word)

        if self.stopwords:
            vocabulary = [word for word in vocabulary if word not in stop_words]

        # Arriving at our number of unique words
        V = len(vocabulary)
        return V


    # Defining our P(Document | Word) function!
    # This function guesses the category based on the doc content
    # Returns the category

    # Avoiding underflow by using np.log of probabilities, not raw probabilities
    # Because algebra: we add the log where we would multiply the raw probability

    # Inputs:
    # doc_content: content of the doc - self.data[self.word_loc]
    # return_posteriors: boolean, saying whether or not you want to print the values
    #      in the posteriors list (per category)
    def classify_doc(self, doc_content, return_posteriors=False):

        # Using count_words function to count the words within the provided doc
        count = self.count_words(doc_content)

        if self.stopwords:
            list_count = list(count.keys())
            for key in list_count:
                if key in stop_words:
                    del count[key]

        categories = []
        posteriors = []

        # This part of our function is putting together the pieces to calculate
        # P(Word | Category):
        
        # The keys of our word_freq dictionary are the categories
        # In other words, we do this one category at a time
        for category in self.word_freqs.keys():
            
            # Finding the default/original probability of that category
            # Here you can see us using the log, rather than the raw probability
            p = np.log(self.p_categories[category])
            
            # The keys of our count dictionary are the words within the document
            for word in count.keys():

                # Numerator for P(Word|Category): Word Frequency in Doc + 1
                num = count[word] + 1
                
                # Denominator for P(Word|Category): Word Frequency Across 
                # All Docs in that Category + Number of Words Across All Docs
                # If the word is not yet in the 'word_freq' dictionary, value = 0
                denom = self.word_freqs[category].get(word, 0) + self.V
                
                # Adding the probability of that word being in the doc based on 
                # category to the probability of the category, in order to arrive
                # at a new and hopefully better probability that the doc
                # is in the current category
                
                # Again, using log instead of raw probability, hence why we are
                # using addition instead of multiplication
                p += np.log(num/denom)
            
            # Adding the current category to the 'categories' list
            categories.append(category)
            
            # Adding the updated probability for the current category to the 
            # 'posteriors' list
            posteriors.append(p)
            
        if return_posteriors:
            print(posteriors)
        
        # Returning the category with the highest probability, for doc_content
        return categories[np.argmax(posteriors)]

    def check_accuracy(self):
        predicted_target = []
        for row in range(len(self.data)):
            predicted_target.append(self.classify_doc(self.data[self.word_loc][row]))
        actual_target = list(self.data[self.category_loc])

        residuals = predicted_target == actual_target
        return residuals.value_counts(normalize=True)