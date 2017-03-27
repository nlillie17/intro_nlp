# coding: utf-8
#Nick, max, austin

#
# hw6 problem 2
#

## Problem 2: Analogies!

def all_words_in_model( wordlist, model ):
    """ returns True if all w in wordlist are in model
        and False otherwise
    """
    for w in wordlist:
        if w not in model:
            return False
    return True


you'll be building:  most_similar

def test_most_similar(model):
    """ example of most_similar """
    print("Testing most_similar on the king - man + woman example...")
    LoM = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)
    # note that topn will be 100 below in check_analogy...
    return LoM




def generate_analogy(word1, word2, word3, model):
    """ solves the analogy word1:word :: word3:? and checks if they are in the model
    """
    
    if all_words_in_model([word1, word2, word3], model) == True:
        LoM = model.most_similar(positive=[word2, word3], negative=[word1], topn=10)
        return LoM[0][0]
    return 0



def check_analogy(word1, word2, word3, word4, model):
    """ checks to make sure that generate_analogy using the first three words returns word4. Depending on where word4 is in the list returns a score. Also makes sure that 
 all words are in the model
    """

    if all_words_in_model([word1, word2, word3, word4], model) == True:
        LoM = model.most_similar(positive=[word2, word3], negative=[word1], topn=100)
        index = 0
        for i in LoM:
            if i[0] == word4:
                score = 100 - index
                return score
            index += 1
        return 0
        
    print("not in model")


#
# Results and commentary...
#

#
# (1) Write generate_analogy and try it out on several examples of your own
#     choosing (be sure that all of the words are in the model --
#     use the all_words_in_model function to help here)
#
# (2) Report two analogies that you create (other than the ones we looked at in class)
#     that _do_ work reaonably well and report on two that _don't_ work well
#     Finding ones that _do_ work well is more difficult! Maybe in 2025, it'll be the opposite (?)

#   analogy1: desert, sand as mountain is to boulders
#   analogy2: city, town as country is to nation
#   analogy3: computer, screen : book, memoir (should have returned something like page or paper)
#   analogy4: basketball, basket : soccer, baskets (should have returned goal)



#
#
# (3) Write check_analogy that should return a "score" on how well word2vec_model
#     does at solving the analogy given (for word4)
#     + it should determine where word4 appears in the top 100 (use topn=100) most-similar words
#     + if it _doens't_ appear in the top-100, it should give a score of 0
#     + if it _does_ appear, it should give a score between 1 and 100: the distance from the
#       _far_ end of the list. Thus, a score of 100 means a perfect score. A score of 1 means that
#       word4 was the 100th in the list (index 99)
#     + Try it out:   check_analogy( "man", "king", "woman", "queen", m ) -> 100
#                     check_analogy( "woman", "man", "bicycle", "fish", m ) -> 0
#                     check_analogy( "woman", "man", "bicycle", "pedestrian", m ) -> 96





#
#
# (4) Create at least five analogies that perform at varying levels of "goodness" based on the
#     check_analogy scoring criterion -- share those (and any additional analysis) with us here!
#
#
#   check_analogy( "red", "blue", "black", "white", model) -> 100
#   check_analogy( "coffee", "tea", "soda", "water", model) -> 0
#   check_analogy( "freedom","democracy","oppression","communism",model) -> 43
#   check_analogy( "war","peace","hate","love",model) -> 100
#
#

