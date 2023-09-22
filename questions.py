import os

import nltk
import sys
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    result_mapping = dict()

    for file_name in os.listdir(directory):
        full_path = os.path.join(directory, file_name)

        with open(full_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            result_mapping[file_name] = file_content

    return result_mapping


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.lower()
    final_list = []
    pre_processed_tokens = nltk.word_tokenize(document)

    # for token in pre_processed_tokens:
    #     if token in nltk.corpus.stopwords.words("english") or token in string.punctuation:
    #         continue
    #
    #     else:
    #         final_list.append(token)
    #
    # return final_list

    for token in pre_processed_tokens:
        if token in nltk.corpus.stopwords.words("english"):
            continue

        only_punctuation = True  # Only filter out those they are purely punctuations and not contains punctuations
        # === , ``, '' appears in tokenized list

        for char in token:
            if char not in string.punctuation:
                only_punctuation = False

        if not only_punctuation:
            final_list.append(token)

    return final_list


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    result_dict = {}

    for document_name in documents:
        lst_of_words = documents[document_name]
        seen_words = set()

        for word in lst_of_words:
            if word not in result_dict and word not in seen_words:
                result_dict[word] = 1
                seen_words.add(word)
            elif word in result_dict and word not in seen_words:
                result_dict[word] += 1
                seen_words.add(word)
            elif word in seen_words:
                continue

    total_documents = len(documents.keys())

    for word in result_dict:
        num_documents_containing_word = result_dict[word]
        result_dict[word] = math.log(total_documents / num_documents_containing_word)

    return result_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf_dict = {}

    for word in query:
        for file in files:
            lst_of_words = files[file]
            num_of_occurrence = lst_of_words.count(word)  # Could be 0
            tf_idf = num_of_occurrence * idfs[word]

            if file in tf_idf_dict:
                tf_idf_dict[file] += tf_idf
            else:
                tf_idf_dict[file] = tf_idf

    sorted_file_list = sorted(list(files.keys()), key=lambda filename: tf_idf_dict[filename], reverse=True)
    return sorted_file_list[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # idf_sentence_dict = {}
    # no_of_query_sentence_dict = {}

    idf_sentence_dict = {sentence: 0 for sentence in sentences}
    no_of_query_sentence_dict = {sentence: 0 for sentence in sentences}

    for word in query:
        for sentence in sentences:
            lst_of_words = sentences[sentence]

            if word in lst_of_words:
                idf_sentence_dict[sentence] += idfs[word]
                no_of_query_sentence_dict[sentence] += 1

    for sentence in no_of_query_sentence_dict:
        query_count = no_of_query_sentence_dict[sentence]
        number_of_words_in_sentence = len(sentences[sentence])
        no_of_query_sentence_dict[sentence] = query_count / number_of_words_in_sentence

    # Using secondary sorting
    sorted_sentences_list = sorted(list(sentences.keys()),
                                   key=lambda s: (idf_sentence_dict[s], no_of_query_sentence_dict[s]), reverse=True)

    return sorted_sentences_list[:n]


if __name__ == "__main__":
    main()
