# coding: utf-8


#
# hw6 problem 3
#

## Problem 3: Paraphrasing!

import textblob 
from textblob import Word
from textblob import TextBlob
 

# A starter function that substitutes each word with it's top match
#   in word2vec.  Your task: improve this with POS tagging, lemmatizing, 
#   and/or at least three other ideas of your own (more details below)
#
def paraphrase_sentence( sentence, model ):
    """ takes in a sentence and for each word replaces it with a similar one based on certain filters
    """
    blob = textblob.TextBlob( sentence )
    print("The sentence's words are")
    LoW = blob.words
    print(LoW)

    NewLoW = []
    for w in LoW:
        if w not in model:
            NewLoW += [w]
        else:
            
            word = advanced_substitution ( w, model )
            #w_alternatives = model.most_similar(positive=[w], topn=100)
            # print("w_alternatives is", w_alternatives)
            
            #first_alternative, first_alternative_score = w_alternatives[0]  # initial one!
            NewLoW += [word]
    
    # you should change this so that it returns a new string (a new sentence),
    # NOT just print a list of words (that's what's provided in this starter code)
    sentence = " ".join(NewLoW)
    return sentence

#
# Function that will make sure the similar word picked doesn't have the same root using the lemmatize function
# will ALSO check to see that it is the same pos
#
def advanced_substitution( word, model):
    '''Takes in a word and checks to make sure the similar word returned does
    not have the same root. Also checks to make sure it has the same POS
    '''
    similar_words = model.most_similar(positive=[word], topn=100) 
    print (similar_words[0])
    PoS_word = simple_POS(TextBlob(word).tags[0][1])
    shortest_length = 100
    shortest_word = ""
    

    lOw = []
    for w in similar_words:
        if w[0][0] != word[0]:                                          #checks to makes sure doesnt start with same letter
            if w[1] > .5:                                               #score cutoff                                                 
                if len(w) < len(word):                                      #makes sure the new paraphrased word is shorter than old word
                    PoS_w = simple_POS(TextBlob(word).tags[0][1])
                    nw = Word(w[0])
                    if nw.lemmatize(PoS_w) != word.lemmatize(PoS_word):     #checks to make sure roots are not the same
                        if PoS_word == PoS_w:                               #checks to make sure words have same part of speech
                            lOw .append(w[0])
    final_word = ""
    for w in lOw:
        length = len(w[0])
        if shortest_length > length:
            shortest_length = length
            final_word = w
    
    return final_word
#
#helper function to return the simple POS 
#
def simple_POS ( pos ):
    '''This function takes in a complex POS tag using tags and returns a simple proxy
    that lemmatize will accept. Lemmatize will default to noun if we don't pass in the 
    correct part of speech
    '''
    if "J" in pos:
        return 'a'
    elif "V" in pos:
        return 'v'
    else:
        return 'n'
# 
# Once the above function is more sophisticated (it certainly does _not_ need to be
#   perfect -- that would be impossible...), then write a file-paraphrasing function:
#
def paraphrase_file(filename, model):
    """ takes in a filename and uses advanced substitution and paraphrase sentence to paraphrase whole file
    """
    data = ""
    with open(filename, 'r') as myfile:
        data=myfile.read().replace('\n','')
        #data=myfile.read()
    
    x = paraphrase_sentence(data,model)
    outFile = open("test_paraphrased.txt", "w")
    outFile.write(x)
    outFile.close()




#
# Results and commentary...
#


# (1) Try paraphrase_sentence as it stands (it's quite bad...)  E.g.,
#         Try:    paraphrase_sentence("Don't stop thinking about tomorrow!", m)
#         Result: ['Did', "n't", 'stopped', 'Thinking', 'just', 'tonight']

#     First, change this so that it returns (not prints) a string (the paraphrased sentence),
#         rather than the starter code it currently has (it prints a list) Thus, after the change:

#         Try:    paraphrase_sentence("Don't stop thinking about tomorrow!", m)
#         Result: "Did n't stopped Thinking just tonight"  (as a return value)

#     But paraphrase_sentence is bad, in part, because words are close to variants of themselves, e.g.,
#         + stop is close to stopped
#         + thinking is close to thinking





# (2) Your task is to add at least three things that improve this performance (though it
#     will necessarily still be far from perfect!) Choose at least one of these two ideas to implement:

#     #1:  Use lemmatize to check if two words have the same stem/root - and _don't_ use that one!
#             + Instead, go _further_ into the similarity list (past the initial entry!)
#     #2:  Use part-of-speech tagging to ensure that two words can be the same part of speech


#       
#     Then, choose two more ideas that use NLTK, TextBlob, or Python strings -- either to guard against
#     bad substitutions OR to create specific substitutions you'd like, e.g., just some ideas:
#        + the replacement word can't have the same first letter as the original
#        + the replacement word is as long as possible (up to some score cutoff)
#        + the replacement word is as _short_ as possible (again, up to some score cutoff...)
#        + replace things with their antonyms some or all of the time
#        + use the spelling correction or translation capabilities of TextBlob in some cool way
#        + use as many words as possible with the letter 'z' in them!
#        + don't use the letter 'e' at all...
#     Or any others you might like!

    '''What we added: lemmatize to make sure does not have the same root, used POS to lemmative verbs or else it wouldn't work properly,
       has a score cutoff of .5, makes sure subsitution word doesn't have same letter if possible and picks the shortest word that fits those criterea'''



# (3) Share at least 4 examples of input/output sentence pairs that your paraphraser creates
#        + include at least one "very successful" one and at least one "very unsuccessful" ones

'''
Example 1 bad: "This is a test file for problem 3. Hopefully this works correctly. It should open file and read into string." ->
'It  a  File  dilemma  hopefully another  properly  ought closed File and write onto rash'

Example 2 bad: "What I like about the RBC trading floor was that it reminded me of a locker room" ->
'How  really  in RBC Trading Floor had it  telling  of a  upstairs'

Example 1 good: "Intellectual professors express knowledge within Literature class" ->
'Intellectual faculty convey expertise Within Poetry Class'

Example 2 good: 'Angry students revolted in response to outrageous homework assignments'->
'Outraged undergraduates enraged  Response to ridiculous homework'
    
'''




# (4) Create a function paraphrase_file that opens a plain-text file, reads its contents,
#     tokenizes it into sentences, paraphrases all of the sentences, and writes out a new file
#     containing the full, paraphrased contents with the word paraphrased in its name, e.g.,
#        + paraphrase_file( "test.txt", model )
#             should write out a file names "test_paraphrased.txt"  with paraphrased contents...
#        + include an example file, both its input and output -- and make a comment on what you
#             chose and how it did! 

'''Chose to paraphrase an expert from Lord of the Rings because Tolkein has been known to be wordy. Used paraphrase file to 
paraphrase text'''

# (Optional EC) For extra-credit (up to +5 pts or more)
#        + [+2] write a function that takes in a sentence, converts it (by calling the function above) and
#          then compares the sentiment score (the polarity and/or subjectivity) before and after
#          the paraphrasing

def compare_attributes(sentence, model):
    '''Takes in a sentence and compares it sentiment score to the paraphrased version 
    sentiment score'''
    y = TextBlob(sentence)
    old_sentiment = y.sentiment
    x = lowest_polarity(sentence, model)
    x = TextBlob(x)
    new_sentiment = x.sentiment

    print("Old sentiment score is ", old_sentiment)
    print("New sentiment score is ", new_sentiment)


#        + [+3 or more beyond this] create another function that tries to create the most-positive or
#          most-negative or most-subjective or least-subjective -- be sure to describe what your
#          function does and share a couple of examples of its input/output...

def lowest_polarity(sentence, model):
    """ finds lowest polarity sentence using advanced advanced_substitution2
    """
    blob = textblob.TextBlob( sentence )
    print("The sentence's words are")
    LoW = blob.words
    print(LoW)

    NewLoW = []
    for w in LoW:
        if w not in model:
            NewLoW += [w]
        else:
            
            word = advanced_substitution2 ( w, model )
            #w_alternatives = model.most_similar(positive=[w], topn=100)
            # print("w_alternatives is", w_alternatives)
            
            #first_alternative, first_alternative_score = w_alternatives[0]  # initial one!
            NewLoW += [word]
    
    # you should change this so that it returns a new string (a new sentence),
    # NOT just print a list of words (that's what's provided in this starter code)
    sentence = " ".join(NewLoW)
    return sentence

def advanced_substitution2( word, model):
    '''Same as old substitution but filters on lowest polarity instead of shortest length
    '''
    similar_words = model.most_similar(positive=[word], topn=100) 
    print (similar_words[0])
    PoS_word = simple_POS(TextBlob(word).tags[0][1])
    shortest_length = 100
    shortest_word = ""
    

    lOw = []
    for w in similar_words:
        if w[0][0] != word[0]:                                          #checks to makes sure doesnt start with same letter
            if w[1] > .5:                                               #score cutoff                                                 
                if len(w) < len(word):                                      #makes sure the new paraphrased word is shorter than old word
                    PoS_w = simple_POS(TextBlob(word).tags[0][1])
                    nw = Word(w[0])
                    if nw.lemmatize(PoS_w) != word.lemmatize(PoS_word):     #checks to make sure roots are not the same
                        if PoS_word == PoS_w:                               #checks to make sure words have same part of speech
                            lOw .append(w[0])
    final_word = ""
    polarity = 0
    lowest_polarity = 1
    for w in lOw:
        x = TextBlob(w)
        polarity = x.sentiment[0]
        if lowest_polarity > polarity:
            lowest_polarity = polarity
            final_word = w
    
    return final_word



