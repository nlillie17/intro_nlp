# Introduction to basic natural language processing
The following is a summary of rudimentary natural language processing computation that I did for class

1. This problem treats words as vectors. Words are orientated spacially in n-dimensional space.  Using word2vec, a neural network that inputs a large corpus of text and produces a vector space, each word is assigned its own corresponding vector in the space. Words that are similar are located in closer proximity. Because of this, you can write analogy functions using basic vector math of addition and subtraction (given the words you are looking for are located in the corpus and thus contained in the vector space). For example, "king - man + women = queen." The following is a snippet of code that uses word2vec to write an analogy engine. The entire code for the problem is found attached as Problem 2. 
 
 #solves the analogy word1:word :: word3:? and checks if they are in the model
   
    
    if all_words_in_model([word1, word2, word3], model) == True:
        LoM = model.most_similar(positive=[word2, word3], negative=[word1], topn=10)
        return LoM[0][0]
    return 0
    

 #checks to make sure that generate_analogy using the first three words returns word4. Depending on where word4 is in the list returns a score. Also makes sure that all words are in the model

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
    
    
 2. This is a naive paraphraser function. Obviously, it is extremely limited but was a good excercise to understand how NLTK, TextBlob and other libraries were set up to facilitate basic NLP. The following is a snippet of code that is the brains of the paraphrasing function. The essance of paraphrasing is to replace complicated words with simpler ones and complex phrases with shorter ones. This function only deals with words not phrases. It takes in one word and returns another that does not have the same root but has the same part of speech, doesn't start with same letter, has a certain score cutoff, and is shorter than original word. 
 
 #Takes in a word and checks to make sure the similar word returned does
    not have the same root. Also checks to make sure it has the same POS
  
    similar_words = model.most_similar(positive=[word], topn=100) 
    print (similar_words[0])
    PoS_word = simple_POS(TextBlob(word).tags[0][1])
    shortest_length = 100
    shortest_word = ""
    

    similar_words = model.most_similar(positive=[word], topn=100) 
    print (similar_words[0])
    PoS_word = simple_POS(TextBlob(word).tags[0][1])
    shortest_length = 100
    shortest_word = ""
    

    lOw = []
    for w in similar_words:
        if w[0][0] != word[0]:                                          
            if w[1] > .5:                                              
                if len(w) < len(word):                                      
                    PoS_w = simple_POS(TextBlob(word).tags[0][1])
                    nw = Word(w[0])
                    if nw.lemmatize(PoS_w) != word.lemmatize(PoS_word):     
                        if PoS_word == PoS_w:                              
                            lOw .append(w[0])
    final_word = ""
    for w in lOw:
        length = len(w[0])
        if shortest_length > length:
            shortest_length = length
            final_word = w
    
    return final_word
    
 3. The final problem is a movie-review sentiment. It attempts to classify movie reviews as positive or negative based on some feature engineering and machine learning. We used decision trees as our algorithim. The main feature engineering can be found below: 
 
 
 feature engineering for movie reviews, dictionary that counts number of positive and negative words in the review. 
            Using, textblob we also check the polarization of words (different gradients of sentiments) and the subjectivity of words 
            (more subjective probably means a more biased review);
            also we tried our hand at doing something similar to context analysis where we check the word before
            the current word and adjust weightings based on the pattern of words (+,-; +,+; -,-; -,+)
    
        # many features are counts!
        positive_count=0.0
        negative_count=0.0
        similar_word_phrase_weight = 2
        prev_word = ''
        for word in movie_reviews.words(fileid):
            w = TextBlob(word)
            if word in pos_set:
                if prev_word in pos_set:
                    positive_count += (1+w.sentiment.polarity*w.sentiment.subjectivity) * similar_word_phrase_weight
                else:
                    positive_count += 1+w.sentiment.polarity*w.sentiment.subjectivity
                   
            elif word in neg_set:
                if prev_word in neg_set:
                    negative_count += (w.sentiment.polarity*w.sentiment.subjectivity-1) * similar_word_phrase_weight
                else:
                    negative_count += w.sentiment.polarity*w.sentiment.subjectivity-1

            prev_word = word


        # here is the dictionary of features...
        features = {'positive': positive_count, 'negative':negative_count}
        return features
