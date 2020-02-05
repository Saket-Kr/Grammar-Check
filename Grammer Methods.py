# Created by SaketKr

# Feature: Grammar checking module
# This application is for checking all possible grammatical errors in a sentence

import nltk
import re
import logging
import spacy
import string
import language_check
import nltk.data
from nltk.stem import WordNetLemmatizer as wnl
from flask import Flask, request, jsonify
from nltk import sent_tokenize
app = Flask(__name__)


class GrammarCheck:

    def __init__(self):
        self.error_list = []
        self.error_dic = dict()
        self.rule_dic = {'i': ['am', 'could', 'should', 'have', 'did', 'had', 'will', 'was', 'can', 'shall', 'may', 'might', 'must', 'would'],
                         'he': ['is', 'could', 'should', 'did',  'has', 'will', 'had', 'was', 'can', 'shall', 'may', 'might', 'must', 'would'],
                         'you': ['are', 'had', 'could', 'should', 'did',  'have', 'will', 'were', 'can', 'shall', 'may', 'might', 'must', 'would']
                         }
        self.tool = language_check.LanguageTool('en-US')
        self.lemmatizer = wnl()
        self.nlp = spacy.load('en')

        logging.basicConfig(filename="log_file.log", format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG)

    def clean_sentence(self, sentence):
        """
        Cleans the sentence to be fed to other functions for looking for the errors
        :param sentence: The sentence to be checked for erros
        :return: the cleaned sentence
        """

        sen_split = sentence.split()

        temp_list = []
        [temp_list.append(i.lower()) if not i.islower() else temp_list.append(i) for i in sen_split]
        if 'customer' in temp_list or 'she' in temp_list or 'it' in temp_list:
            sentence = ' '.join('He' if x in ['Customer', 'She', 'It'] else 'he' if x in ['customer', 'she', 'it'] else x for x in sen_split)

        if 'they' in temp_list or 'we' in temp_list:
            sentence = ' '.join('you' if x in ['they', 'we'] else 'You' if x in ['They', 'We'] else x for x in sen_split)

        return sentence

    def end_with_punctuation(self, sentence):
        """
        All sentences should end with some punctuation.
        :param sentence: Sentence to be parsed.
        """
        if not re.match(r'[\.?!]$', sentence[-1]):
            self.error_list.append("Every sentence should end with either of '.', '?' or '!'.")

    def noun_capitalise(self, sentence):
        """
        Method for capitalisation of Nouns
        :param sentence: sentence to be checked for errors
        """
        sen_split = sentence.split()
        for word in sen_split:
            # removing the punctuation from the word extracted.
            word = word.translate(str.maketrans('', '', string.punctuation))
            if self.nlp(word)[0].tag_ in ['NNP', 'NNPS']:
                if word[0] != word[0].upper():
                    self.error_list.append("The noun '" + word.strip('.') + "' should be capitalised.")

    def hyphen_space(self, sentence):
        """
        Method for checking spaces before '-'. There shouldn't be any whitespace before or after '-'
        :param sentence: sentence to be checked for the error
        :return: None
        """
        if '-' in sentence:
            if sentence[-1] == '-':
                self.error_list.append('Sentence should not end with "-".')
            elif sentence[sentence.find('-') - 1] == ' ' or sentence[sentence.find('-') + 1] == ' ':
                self.error_list.append("There shouldn't be any spaces before or after the '-' symbol.")

    def check_for_i(self, sentence):
        """
        Method for 'I'
        :param sentence: sentence to be checked for errors
        :return: None
        """
        if sentence[-1] == '?':
            return

        # change sentence to lower case and split and save to list
        sen_split = sentence.lower().split()

        # find the pos tags of the sentence using nltk library
        text = nltk.word_tokenize(sentence)

        if sen_split[0] == 'i':

            # checking each of the sentences with the rules
            if 'i' in sen_split:

                # word next to 'I'
                if len(sen_split) - sen_split.index("i") > 1:
                    next_word = sen_split[sen_split.index("i") + 1]  # the word next to 'I'

                    # pos tag of the word next to 'I'
                    next_word_tag = self.nlp(next_word)[0].tag_  # the postag of the next word

                    # the word after 'I have/had' should be like 'played/eaten/it/said'
                    if len(sen_split) - sen_split.index("i") >= 3:
                        if next_word in ['have', 'had']\
                                and sen_split[sen_split.index("i") + 2] not in ['not']\
                                and self.nlp(sen_split[sen_split.index("i") + 2])[0].tag_ not in ['VBD', 'VBN', 'DT', 'RB']:
                            self.error_list.append("With 'I have/had', second form of the verb should be used ("\
                                                   + sen_split[sen_split.index("i") + 2].strip('.')+ "), like played, gone.")

                    """After 'I have been' there should be 'ing' with the verb."""
                    if len(sen_split) - sen_split.index("i") >= 4:
                        if next_word == 'have'\
                           and sen_split[sen_split.index("i") + 2] == 'been'\
                           and sen_split[sen_split.index("i") + 3][-3:] != 'ing'\
                           and self.nlp(sen_split[sen_split.index("i") + 3])[0].tag_ not in ['JJ', 'VBG', 'RB', 'IN', 'UH', 'VBN', 'VBD']:
                            self.error_list.append("With 'I have been', the verb form should be past tense\
                             or present participle ("+ sen_split[sen_split.index("i") + 3].strip('.')+"), like playing, gone." )

                    if next_word == 'been':
                        self.error_list.append("'been' cannot come after 'I'.")

                    if next_word[-3:] == 'ing':
                        self.error_list.append("Present participle form of the verb shouldn't be used ("+ next_word + ").")

                    """checking for given such examples: I [work (NN)], I [do (VB)], I [worked (VBN)], I [walked (VBD)].
                        Checking for the the second word here. """
                    if next_word_tag not in ['NN', 'VBN', 'VB', 'VBD', 'NNS', 'VBP', 'JJ', 'IN', 'UH'] and next_word not in self.rule_dic['i']:
                        self.error_list.append("Wrong usage of 'I'. 'I' should be used with a verb (work, play, etc.) or modals (would, could, etc).")

                    """ checking length of sentence after 'I' to be greater than two. """
                    if len(sen_split) - sen_split.index("i") > 2:

                        """ If the first word is 'I', the third and fourth words are 'have' and 'been' respectively, 
                             the second word must be would/could/should."""
                        if sen_split[sen_split.index("i") + 2] == 'have'\
                                and sen_split[sen_split.index("i") + 3] == 'been'\
                                and self.nlp(next_word)[0].tag_ != "MD":
                            self.error_list.append('You should use modals (would, could, etc.) after the Pronoun here.')

                        """ If the first word is 'I' & the third word is 'been', the second word should be have/had."""
                        if sen_split[sen_split.index("i") + 2] == 'been'\
                                and next_word not in ['have', 'had']:
                            self.error_list.append("You should use have or had after the pronoun here.")

                    if len(sen_split) - sen_split.index("i") >= 3:

                        """Checking for determiners before a noun."""
                        if next_word in ['am', 'was']\
                           and self.nlp(sen_split[sen_split.index("i") + 2])[0].tag_ in ['NN', 'JJS' 'NNP']:
                           self.error_list.append("You should use a determiner (a, an, the, this, etc) before the Noun \
                           or Superlative adjective.")

                        """Checking sentences like I am reading, I am playing"""
                        if next_word in ['am', 'was']\
                                and self.nlp(sen_split[sen_split.index("i") + 2])[0].tag_ \
                                not in ['NN', 'RB', 'JJ', 'VBN', 'IN', 'DT', 'NNP', 'VBP', 'PRP$']\
                                and sen_split[sen_split.index("i") + 2][-3:] != 'ing':
                            self.error_list.append("Present or past participle form of the verb should be used ("\
                                                   + sen_split[sen_split.index("i") + 2].strip('.')+"), like playing, gone, etc.")

                        if len(sen_split) - sen_split.index("i") >= 5:

                            """Word after 'I have been' should have 'ing' or like, 'dead/told/made' """
                            if sen_split[sen_split.index(next_word) + 2][-3:] != 'ing'\
                                    and self.nlp(sen_split[sen_split.index(next_word) + 3])[0].tag_ not in ['JJ', 'VBN', 'VBD']\
                                    and sen_split[sen_split.index("i") + 1] == 'have'\
                                    and sen_split[sen_split.index("i") + 2] == 'been':
                                self.error_list.append("With sentence formations like 'I have been' we use present or past participle \
                                form of the verb ("+sen_split[sen_split.index(next_word) + 3].strip('.')\
                                                       +") like, playing, done, gone, etc.")

                        # if the word after 'I' is would, could or should
                        if self.nlp(next_word)[0].tag_ == "MD":
                            """checking the word after would, could or should, it should be, I would sleep, I would not pee or 
                            I would 'have', I would be"""
                            if self.nlp(sen_split[sen_split.index(next_word) + 1])[0].tag_ not in ['NN', 'VBG', 'RB', 'VB']\
                                    and sen_split[sen_split.index(next_word) + 1] not in ['have', 'not']:
                                self.error_list.append("After 'I would', verb or adverb should be used like do, play, go, etc.")

                        if len(sen_split) - sen_split.index("i") >= 5:
                            """Word after 'I would have been' should have 'ing' """
                            if sen_split[sen_split.index(next_word) + 3][-3:] != 'ing'\
                                    and self.nlp(sen_split[sen_split.index(next_word) + 3])[0].tag_ not in ['VBG', 'JJ', 'VBN', 'VBD']\
                                    and sen_split[sen_split.index("i") + 2] == 'have'\
                                    and sen_split[sen_split.index("i") + 3] == 'been':
                                self.error_list.append("With sentence formations like 'I would have been' we use present\
                                 or past participle form of the verb ("+sen_split[sen_split.index(next_word) + 3].strip('.')\
                                                       + ") like, playing, gone, etc.")

    def check_for_he(self, sentence):
        """
        Method for 'He'
        :param sentence: sentence to be checked for errors
        :return: None
        """
        if sentence[-1] == '?':
                return

        # change sentence to lower case and split and save to list
        sen_split = sentence.lower().split()

        # find the pos tags of the sentence using nltk library
        text = nltk.word_tokenize(sentence)

        if sen_split[0] == 'he':

            if 'he' in sen_split and len(sen_split) - sen_split.index("he") > 1:
                # word next to 'he'
                next_word = sen_split[sen_split.index("he") + 1]  # the word next to 'He'

                # pos tag of the word next to 'he'
                next_word_tag = self.nlp(next_word)[0].tag_  # the postag of the next word

                if next_word == 'been':
                    self.error_list.append("'been' should not be used here.")

                if next_word[-3:] == 'ing':
                    self.error_list.append("Present participle form of the verb shouldn't be used with the pronoun ("
                                           + next_word + "), like playing, singing, etc.")

                """checking for given such examples: He [works (NNS)], He [does (VBZ)], He [worked (VBN)], He [walked (VBD)].
                    Checking for the the second word here. """
                if next_word_tag not in ['NNS', 'VBZ', 'VBD', 'VBN', 'IN'] and next_word not in self.rule_dic['he']:
                    self.error_list.append("Pronoun should be used with a third form of verb like plays, works, etc,or modals like would, could, should, etc. (" + next_word+")")

                if len(sen_split) - sen_split.index("he") >= 2:
                    if next_word in ['has', 'had'] and sen_split[sen_split.index("he") + 2][-3:] != 'ing':
                        self.error_list.append("Second form of the verb should be used with has/had ("\
                                               + sen_split[sen_split.index("he") + 2].strip('.') + ") like plays, works, etc.")

                    """ checking length of sentence after 'he' to be greater than two. """
                if len(sen_split) - sen_split.index("he") >= 4:

                    """ If the first word is 'He', the third and fourth words are 'have' and 'been' respectively, 
                         the second word must be would/could/should."""
                    if sen_split[sen_split.index("he") + 2] == 'have'\
                            and sen_split[sen_split.index("he") + 3] == 'been'\
                            and self.nlp(next_word)[0].tag_ != "MD":
                        self.error_list.append('You should use modals after the pronoun like would, could, etc.')

                    """ If the first word is 'he' and the third word is 'been', the second word should be has/had."""
                    if sen_split[sen_split.index("he") + 2] == 'been' and next_word not in ['has', 'had']:
                        self.error_list.append("One should use has or had.")

                    if len(sen_split) - sen_split.index("he") >= 4:
                        """Word after 'He has been' should have 'ing' or 'sick'"""
                        if sen_split[sen_split.index(next_word) + 2][-3:] != 'ing'\
                                and self.nlp(sen_split[sen_split.index(next_word) + 2])[0].tag_ \
                                not in ['JJ', 'VBG', 'RB', 'IN', 'UH', 'VBD', 'VBN']\
                                and sen_split[sen_split.index("he") + 1] in ['has', 'had']\
                                and sen_split[sen_split.index("he") + 2] == 'been':
                            self.error_list.append("Wrong form of the verb is used here ("\
                                                   + sen_split[sen_split.index(next_word) + 2].strip('.') + ").")

                    # if the word after 'He' is would, could or should
                    if len(sen_split) - sen_split.index("he") >= 3:

                        """Checking for determiners before a noun."""
                        if next_word in ['is', 'was']\
                                and self.nlp(sen_split[sen_split.index("he") + 2])[0].tag_ in ['NN', 'JJS', 'NNP']:
                            self.error_list.append("You should use a determiner before the Noun \
                            or Superlative adjective like a, an, the, this, etc.")

                        """Checking sentences like He is/was reading, He is/was playing"""
                        if next_word in ['is', 'was']\
                                and self.nlp(sen_split[sen_split.index("he") + 2])[0].tag_ \
                                not in ['JJ', 'VBG', 'UH', 'JJR', 'RB', 'PRP', 'PRP$', 'DT', 'IN', 'VBP', 'NN']\
                                and sen_split[sen_split.index("he") + 2][-3:] != 'ing':
                            self.error_list.append("The present or past participle form of the verb should be used ("
                                                   + sen_split[sen_split.index("he") + 2].strip('.') + ') like playing, done, etc.')

                        if self.nlp(next_word)[0].tag_ == "MD":

                            """checking the word after would, could or should, it should be, He would sleep, He would not pee or 
                            He would 'have', He would be"""
                            if self.nlp(sen_split[sen_split.index(next_word) + 1])[0].tag_ not in ['NN', 'VB', 'IN']\
                                    and sen_split[sen_split.index(next_word) + 1] not in ['have', 'not']:
                                self.error_list.append("After a pronoun followed by 'would', a verb or an adverb should be used ("
                                                       + sen_split[sen_split.index(next_word) + 1].strip('.')
                                                       + ") like sleep, see, etc.")

                            if len(sen_split) - sen_split.index("he") >= 5:
                                """Word after 'He would have been' should have 'ing' """
                                if sen_split[sen_split.index(next_word) + 3][-3:] != 'ing'\
                                        and self.nlp(sen_split[sen_split.index(next_word) + 3])[0].tag_ \
                                        not in ['JJ', 'VBG', 'RB', 'VBN', 'VBD']\
                                        and sen_split[sen_split.index("he") + 2] == 'have'\
                                        and sen_split[sen_split.index("he") + 3] == 'been'\
                                        and self.nlp(next_word)[0].tag_ == "MD":
                                    self.error_list.append("There is some mistake after the noun/pronoun and 'would have been' ("
                                                           + sen_split[sen_split.index(next_word) + 3].strip('.') + ").")

    def check_for_you(self, sentence):
        """
        Method for 'You'
        :param sentence: sentence to be checked for errors
        :return: None
        """
        if sentence[-1] == '?':
            return

        # change sentence to lower case and split and save to list
        sen_split = sentence.lower().split()

        # find the pos tags of the sentence using nltk library
        text = nltk.word_tokenize(sentence)

        if sen_split[0] == 'you':

            if 'you' in sen_split and len(sen_split) - sen_split.index("you") > 1:
                # word next to 'you'
                next_word = sen_split[sen_split.index("you") + 1]  # the word next to 'you'

                # pos tag of the word next to 'you'
                next_word_tag = self.nlp(next_word)[0].tag_  # the postag of the next word

                # check if 'been' is there after 'you' which is wrong
                if next_word == 'been':
                    self.error_list.append("'been' cannot come after the pronoun/noun.")

                # check if the word after 'they' has 'ing' before it, i.e., it's a verb (playing).
                if next_word[-3:] == 'ing':
                    self.error_list.append("Present participle form of the verb should NOT be used after the "
                                           "pronoun/noun ( "
                                           + next_word + ") like play, work.")

                """checking for given such examples: you [work (NN)], you [do (VB)], you [walked (VBD)].
                    Checking for the the second word here. """
                if next_word_tag not in ['NN', 'VB', 'VBD', 'VBP', 'VBN', 'IN'] and next_word not in self.rule_dic['you']:
                    self.error_list.append("Pronouns/nouns should be used with a verb or modals ("
                                           + next_word + ") like you love, you sing.")

                """ checking length of sentence after 'you' to be greater than two. """
                if len(sen_split) - sen_split.index("you") >= 3:

                    """ If the first word is 'you', the third and fourth words are 'have' and 'been' respectively, 
                         the second word must be would/could/should."""
                    if sen_split[sen_split.index("you") + 2] == 'have'\
                            and sen_split[sen_split.index("you") + 3] == 'been'\
                            and self.nlp(next_word)[0].tag_ != "MD":
                        self.error_list.append('You should use modals like would, could, etc here.')

                    """ If the first word is 'you' and the third word is 'been', the second word should be has/had."""
                    if sen_split[sen_split.index("you") + 2] == 'been'\
                            and next_word not in ['have', 'had']:
                        self.error_list.append("You should use have or had after the pronoun.")

                    """Checking sentences like 'you are reading, you are playing"""
                    if next_word in ['are', 'were']\
                            and self.nlp(sen_split[sen_split.index("you") + 2])[0].tag_ \
                            not in ['JJ', 'VBG', 'UH', 'JJR', 'IN', 'VBN', 'RB', 'NNP', 'RB', 'DT', 'JJS']\
                            and sen_split[sen_split.index("you") + 2][-3:] != 'ing':
                        self.error_list.append("The present or past participle form of the verb should be used ("
                                               + sen_split[sen_split.index("you") + 2].strip('.') + ") like reading, "
                                                                                                    "gone, etc.")

                    if len(sen_split) - sen_split.index("you") >= 4:
                        """Word after 'you have been' should have 'ing' """
                        if sen_split[sen_split.index(next_word) + 2][-3:] != 'ing'\
                                and self.nlp(sen_split[sen_split.index(next_word) + 2])[0].tag_ \
                                not in ['JJ', 'VBG', 'RB', 'IN', 'UH']\
                                and sen_split[sen_split.index("you") + 1] in ['have', 'had']\
                                and sen_split[sen_split.index("you") + 2] == 'been':
                            self.error_list.append("With sentence formations like 'you have been' we use present or "
                                                   "past participle form of the verb (" + sen_split[sen_split.index\
                                                    (next_word) + 2].strip('.') + ") like gone, singing.")

                    # if the word after 'you' is would, could or should
                    if len(sen_split) - sen_split.index("you") >= 3:
                        if self.nlp(next_word)[0].tag_ == "MD":
                            """checking the word after would, could or should, it should be, you would sleep, 
                            you would not pee or you would 'have', you would be """
                            if self.nlp(sen_split[sen_split.index(next_word) + 1])[0].tag_ not in ['NN', 'VB']\
                                    and sen_split[sen_split.index(next_word) + 1] not in ['have', 'not']:
                                self.error_list.append("After 'you would', 'a noun, verb or adverb' is used ("\
                                                       + sen_split[sen_split.index(next_word) + 1].strip('.') + ").")

                            if len(sen_split) - sen_split.index("you") >= 5:
                                """Word after 'you would have been' should have 'ing' """
                                if sen_split[sen_split.index(next_word) + 3][-3:] != 'ing'\
                                        and self.nlp(sen_split[sen_split.index(next_word) + 3])[0].tag_ not in ['JJ', 'VBG']\
                                        and sen_split[sen_split.index("you") + 2] == 'have'\
                                        and sen_split[sen_split.index("you") + 3] == 'been'\
                                        and self.nlp(next_word)[0].tag_ == "MD":
                                    self.error_list.append("With sentence formations like 'you would have been'"
                                                           "we use present or past participle form of the verb ("
                                                           + sen_split[sen_split.index(next_word) + 3].strip
                                                               ('.') + ") like told, singing, gone, etc.")

    def using_grammar_check(self, sentence):
        """
        Using the Language Check Tool
        :param sentence: sentence to be checked for errors
        :return: None
        """
        """Using the language check for first suggestions."""
        matches = self.tool.check(sentence)
        if len(matches) > 0:
            for i in range(len(matches)):
                if matches[i].msg not in ['Possible typo: you repeated a whitespace', 'Add a space between sentences', 'Possible spelling mistake found']:
                    self.error_list.append(matches[i].msg)

    def etcetera_check(self, sentence):
        """
        Method for usage of etcetera.
        :param sentence: sentence to be checked for errors
        :return: None
        """
        pattern = '\s?[a-z]+\s?,[,|\s|.]*'
        count = len(re.findall(pattern, sentence))
        if count == 1\
                and self.nlp(re.findall(pattern, sentence)[0].split(',')[0].strip('.'))[0].tag_ in ['NN', 'NNS', 'NNP', 'NNPS']\
                and 'etc' not in sentence\
                and 'and' not in sentence:
            self.error_list.append("Should have used 'and' here.")

        if count > 1\
                and self.nlp(re.findall(pattern, sentence)[0].split(',')[0].strip('.'))[0].tag_ in ['NN', 'NNS', 'NNP', 'NNPS']\
                and 'etc' not in sentence\
                and 'and' not in sentence:
            self.error_list.append('Should use "et cetera" between multiple nouns.')

    def grammar_check(self, sentence):
        """
        Method to integrate all of the error detection functions and then retuning the error dictionary with keys
         as the sentence and values as a list of all the errors found.
        :param sentence: sentence to be checked for errors
        :return: The error dictionary
        """
        self.error_dic = dict()
        sent = sent_tokenize(sentence)
        for s in sent:
            self.error_list = []
            self.using_grammar_check(s)
            sen = self.clean_sentence(s)
            try:
                self.check_for_you(sen)
                self.check_for_he(sen)
                self.check_for_i(sen)
                self.noun_capitalise(sen)
                # self.end_with_punctuation(sen)
                self.hyphen_space(sen)

            except Exception as e:
                logging.error("{} - {}".format(s, str(e)))
            logging.info("{} - {}".format(s,  ', '.join(self.error_list)))
            self.error_dic[s] = self.error_list

        return self.error_dic


a = GrammarCheck()


# Method to integrate all of the functions together.
@app.route('/gramcheck/', methods=['POST'])
def gram_check():
    content = request.json
    sentence = content['text']

    suggestions = a.grammar_check(sentence)
    return jsonify({"response": suggestions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
