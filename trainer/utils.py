import sys
import csv
import pandas as pd
import gensim
import smart_open

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class Constants:

    QUERY_FILE_ADDRESS = "/home/fsarvi/codes/matchzoo_tests/data/train-queries.csv"
    PRODUCT_FILE_ADDRESS = "/home/fsarvi/codes/matchzoo_tests/data/products.csv"
    DERIVED_DATA_PREFIX_ADDRESS = "/home/fsarvi/codes/matchzoo_tests/data/derived_data/"
    ALL_PRODUCTS_NAME_CLEAN_CORPUS = "/home/fsarvi/codes/matchzoo_tests/data/derived_data/all_product_name_tokens.csv"
    ALL_QUERIES_CLEAN_CORPUS = "/home/fsarvi/codes/matchzoo_tests/data/derived_data/all_query_tokens.csv"
    TRAIN_SET_ITEM_QUERY_TOKENS_PLUS_LABELS_ADDRESS = "/home/fsarvi/codes/matchzoo_tests/data/derived_data/train_set_query_product_tokens_and_ids_with_label.csv"

    REDUCED_WORD2VEC_MODEL_TRAINED_ON_THE_WHOLE_CORPUS_ADDRESS = "/home/fsarvi/codes/matchzoo_tests/data/derived_data/reduced_word2vec_model_trained_on_whole_corpus.csv"

    WORD2VEC_MODEL_TRAINED_ON_THE_WHOLE_CORPUS_ADDRESS = "/home/fsarvi/codes/matchzoo_tests/trained_models/word2vec/word2vec_model_trained_on_whole_corpus"


class Utils:

    def __init__(self):
        pass

    def save_list_to_csv(self, data, output_file_name):
        if not isinstance(data, list):
            print("input must be a list")
            return
        if not isinstance(data[0], list):
            with open(output_file_name, 'w') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerows(map(lambda x: [x], data))
        else:
            with open(output_file_name, "w") as f:
                writer = csv.writer(f)
                writer.writerows(data)

    def write_dict_to_csv_with_a_row_for_each_key(self, data, output_file_name):
        with open(output_file_name, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in data.items():
                writer.writerow([key, value])

    def read_csv_to_dict_with_first_column_as_keys(self, file_name):
        with open(file_name) as csv_file:
            reader = csv.reader(csv_file)
            mydict = dict(reader)

        return mydict

    def number_list_from_str_list(self, str_list):
        return [float(i) for i in str_list[1:len(str_list) - 1].split(',')]

    def read_csv_to_pandas_dataframe(self, file_name, sep=','):
        data_df = pd.read_csv(file_name, sep=sep)
        return data_df

    def read_corpus(self, fname, tokens_only=False):
        with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                if tokens_only:
                    yield gensim.utils.simple_preprocess(line)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

    def gensim_text_cleaning_and_spliting(self, text):
        return gensim.utils.simple_preprocess(text)

    def read_space_delimeter_csv_file(self, file_name):
        with open(file_name) as f:
            file = [line.split() for line in f]
        return file

    def read_csv_file_line_by_line(self, file_name):
        with open(file_name) as f:
            data = f.readlines()
        return data

    def convert_list_of_string_lists_to_list_of_numberic_lists(self, data):
        # e.x. ['1,2,3', '3,3,4', '5,2,1'] -> [[1,2,3],[3,3,4],[5,2,1]]
        new_feature_list = []
        for indx, row in enumerate(data):
            new_row = row.split(',')
            new_row = [float(i) for i in new_row]
            new_feature_list.append(new_row)

        return new_feature_list

    def read_text_file_in_lines(self, file_address):
        examples = []
        with open(file_address,
                  errors='ignore') as f:
            examples += [line for line in f]
        return examples

if __name__ == "__main__":
    train_set = pd.read_csv("/Users/fsarvi/PycharmProjects/CIKM_open_data_experiments/data/derived_data/train_set_itemId_queryId_Label.csv")

    pass
