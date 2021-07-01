import tensorflow as tf
from datetime import datetime

TAG_learningrate='Custom learning rate'

def setupTensorboardWriterforLR(path):
    logdir = path+"/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
                                                    
def build_learning_rate(name):
    if name == 'fixedstep':
        return tf.keras.callbacks.LearningRateScheduler(lambda epoch: fixedsteplearningratefn(epoch), verbose=True)
    elif name == 'fixed':
        return tf.keras.callbacks.LearningRateScheduler(fixedlearningratefn, verbose=0)
    elif name == 'warmupexpdecay':
        return tf.keras.callbacks.LearningRateScheduler(warmupexpdecaylrfn, verbose=0)

def fixedlearningratefn(epoch):
    #Inside the learning rate function, use tf.summary.scalar() to log the custom learning rate.
    lr = 0.0010000000474974513
    tf.summary.scalar(TAG_learningrate, data=lr, step=epoch)
    return lr

def fixedsteplearningratefn(epoch):
    lr = 0.01
    if epoch < 3:
        lr = 1e-3
    elif epoch >= 3 and epoch < 7:
        lr = 1e-4
    else:
        lr = 1e-5
    #Inside the learning rate function, use tf.summary.scalar() to log the custom learning rate.
    tf.summary.scalar(TAG_learningrate, data=lr, step=epoch)
    return lr

def warmupexpdecaylrfn(epoch):
    start_lr = 0.00001
    min_lr = 0.00001
    max_lr = 0.00002 #* strategy.num_replicas_in_sync
    rampup_epochs = 7
    sustain_epochs = 0
    exp_decay = .8
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
        #Inside the learning rate function, use tf.summary.scalar() to log the custom learning rate.
        tf.summary.scalar(TAG_learningrate, data=lr, step=epoch)
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)