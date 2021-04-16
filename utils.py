def plot(writer, epoch_id, monitored_quantities_train,
         monitored_quantities_test):

    for k, v in monitored_quantities_train.items():
        if type(v) == list:
            for ind, elem in enumerate(v):
                writer.add_scalar(f'{k}_{ind}/train', elem, epoch_id)
        else:
            writer.add_scalar(f'{k}/train', v, epoch_id)

    for k, v in monitored_quantities_test.items():
        if type(v) == list:
            for ind, elem in enumerate(v):
                writer.add_scalar(f'{k}_{ind}/test', elem, epoch_id)
        else:
            writer.add_scalar(f'{k}/test', v, epoch_id)