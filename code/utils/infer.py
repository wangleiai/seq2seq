from keras.models import Model, load_model
from keras.layers import Input,LSTM,Dense
from keras import callbacks
import numpy as np
import process_data
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

model = load_model('en_to_cn.h5')
model.summary()

latent_dim = 126 # LSTM 的单元个数
batch_size = 8
en_vocab_size = 3000
cn_vocab_size = 3000


# 编码器的输入
encoder_inputs = Input(shape=(None, en_vocab_size), name='input_1')
#编码器要求返回状态,
encoder = LSTM(latent_dim, return_state=True, name='lstm_1')
# 调用编码器，得到编码器的输出，以及状态信息
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#丢弃outputs
encoder_state = [state_h, state_c]
# print('encoder state')
# print(encoder_state)
# 解码器的输入
decoder_inputs = Input(shape=(None, cn_vocab_size))
decoder_infer_state_h = Input(shape=(latent_dim,))
decoder_infer_state_c = Input(shape=(latent_dim,))
decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]
# 建立解码器
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='lstm_2')
#将编码器输出的状态作为初始解码器的初始状态
deocer_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=decoder_infer_state)
decoder_out_state = [decoder_state_h, decoder_state_c]
# print(decoder_out_state)

# 添加全连接层
decoder_dense = TimeDistributed(Dense(cn_vocab_size, activation='softmax'), name='time_distributed_1')
decoder_outpusts = decoder_dense(deocer_outputs)

encoder_model = Model(encoder_inputs, encoder_state)
encoder_model.load_weights('en_to_cn.h5', by_name=True)
# print('encoder model struct')
# encoder_model.summary()

deocder_model = Model([decoder_inputs] + decoder_infer_state,
                      [decoder_outpusts] + decoder_out_state)

for layer in deocder_model.layers:
    if layer.name=='time_distributed_1':
        layer.set_weights = load_model('en_to_cn.h5').get_layer('time_distributed_1').get_weights()
        # print('success ')
    if layer.name=='lstm_2':
        layer.set_weights = load_model('en_to_cn.h5').get_layer('lstm_2').get_weights()

deocder_model.summary()


# 加载数据集
en_dict, cn_dict = process_data.get_dict('../cmn.txt', max_size=3000)
cn_re_dict = dict(zip(cn_dict.values(), cn_dict.keys()))
en_sen = ['I wish I could help you.']

en_sen = process_data.en_tokenize(en_sen)
en_sen = process_data.add_eos_sos(en_sen)
# print(en_sen)
en_sen = process_data.word_to_idx(en_sen, en_dict)
# print(en_sen)
en_sen = process_data.pad_unk(en_sen)
en_sen = process_data.idx_to_onehot(en_sen, cates=len(en_dict))
# print(en_sen)


# 使用encoder model 进行预测
states = encoder_model.predict(np.array(en_sen))

length = 0
cn_input = 'sos'
cn_out = ''
while length<=20:
    cn_idx = cn_dict[cn_input]
    cn_onehot = to_categorical(cn_idx, len(cn_dict))
    cn_onehot = np.expand_dims(cn_onehot, 0)
    cn_onehot = cn_onehot[np.newaxis,:]
    outputs, state_h, state_c = deocder_model.predict([cn_onehot, states[0], states[1]])
    # print(np.array(outputs).shape)
    states = [state_h, state_c]
    length += 1
    print(outputs)
    print(outputs.shape)
    print(np.argmax(outputs, 2))
    print(cn_re_dict[np.argmax(outputs, 2)[0][0]])
    # maxn = 0
    # outputs = list(outputs)
    # for i in range(len(outputs[0])):
    #     if outputs[0][i]>maxn:
    #         maxn = outputs[0][i]
    #     print(outputs[0][i])
    # print(maxn)
    break
    # idx = np.argmax()


