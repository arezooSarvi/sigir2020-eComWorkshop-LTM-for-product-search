import sys

import matchzoo as mz
from trainer.utils import Utils
import time
import keras


class Unbuffered(object):

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)


TRAIN_FILE = "../../data/train"
TEST_FILE = "../../data/test"
VALID_FILE = "../../data/dev"
TEST_RUN_FILE = "../../data/rt-polarity.test"


def generate_test_sample_index():

    test_data = []
    with open(TEST_RUN_FILE,
              errors='ignore') as f:
        test_data += [line for line in f]

    item_query_ids = []
    for test_sample in test_data:
        try:
            item_query_id = test_sample.split()[0]
            item_query_ids.append(item_query_id)
        except:
            pass
            print("error in ** generate_test_sample_index ** ")
    return item_query_ids


def write_predictions_in_test_score_file(test_set_predictions, item_query_ids, best=False):
    result = {}
    for id, score in enumerate(test_set_predictions):
        result[item_query_ids[id]] = score[0]
    post_fix = "_last_model"
    if best:
       post_fix = "_best_model"
    Utils().write_dict_to_csv_with_a_row_for_each_key(result, "features" + post_fix + ".csv")


item_query_ids = generate_test_sample_index()

cikm_train_data = mz.load_data_pack(TRAIN_FILE)
cikm_test_data = mz.load_data_pack(TEST_FILE)
cikm_validation_data = mz.load_data_pack(VALID_FILE)
cikm_train_data.shuffle(inplace=True)


preprocessor = mz.preprocessors.DSSMPreprocessor()
preprocessor.fit(cikm_train_data)

print(preprocessor.context)
train_processed = preprocessor.transform(cikm_train_data)
test_processed = preprocessor.transform(cikm_test_data)
validation_processed = preprocessor.transform(cikm_validation_data)

ranking_task = mz.tasks.Ranking()
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=25),
    mz.metrics.MeanAveragePrecision()
]


model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 144
model.params['mlp_num_fan_out'] = 124
model.params['mlp_activation_func'] = 'relu'
model.params['optimizer'] = 'adam'	
model.guess_and_fill_missing_params()
print(model.params)

model.build()
model.compile()
model.backend.summary()

test_x, test_y = test_processed[:].unpack()
val_x, val_y = validation_processed[:].unpack()

evaluate = mz.callbacks.EvaluateAllMetrics(model, x=val_x, y=val_y, batch_size=len(val_y))

data_generator = mz.PairDataGenerator(train_processed, num_dup=2, num_neg=2, batch_size=128)
callback_earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=100, min_delta=0.001)
mcp_save = keras.callbacks.ModelCheckpoint('best_one_dssm', save_best_only=True, monitor='val_loss', mode='min')

start_time = time.time()
model.fit_generator(data_generator, epochs=1000, validation_data=(val_x, val_y), callbacks=[evaluate, callback_earlystopping, mcp_save], verbose=2)

print("===================================================== Training time =====================================================")
print("--- %s seconds ---" % (time.time() - start_time))
print("=========================================================================================================================")

start_time = time.time()
test_set_predictions = model.predict(test_x)

print("===================================================== Inference time =====================================================")
print("--- %s seconds ---" % (time.time() - start_time))
print("=========================================================================================================================")
try:
    model.save('DSSM-model')
except:
    pass
write_predictions_in_test_score_file(test_set_predictions, item_query_ids)

# ---------------------
model.backend.load_weights("best_one_dssm", by_name=True)
test_set_predictions = model.predict(test_x)
write_predictions_in_test_score_file(test_set_predictions, item_query_ids, best=True)

print("end of program")
