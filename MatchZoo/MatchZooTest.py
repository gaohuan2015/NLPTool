import matchzoo as mz

train_pack = mz.datasets.snli.load_data(
    'train', task='classification', return_classes=True)
valid_pack = mz.datasets.snli.load_data(
    'dev', task='classification', return_classes=True)
predict_pack = mz.datasets.snli.load_data(
    'test', task='classification', return_classes=True)

preprocessor = mz.preprocessors.BasicPreprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)
predict_processed = preprocessor.transform(predict_pack)

ranking_task = mz.tasks.Classification(num_classes=2)
ranking_task.metrics = [mz.metrics.Precision()]

model = mz.models.MVLSTM()

model.params['task'] = ranking_task
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'
model.build()
model.compile()

train_generator = mz.PairDataGenerator(
    train_processed, num_dup=1, batch_size=64, shuffle=True)

valid_x, valid_y = valid_processed.unpack()
test_x, test_y = predict_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(
    model, x=valid_x, y=valid_y, batch_size=64)

history = model.fit_generator(
    train_generator,
    epochs=10,
    callbacks=[evaluate],
    workers=5,
    use_multiprocessing=False)
result = model.predict(predict_pack)
print(result)
