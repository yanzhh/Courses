from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)  # 远程下载数据
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


filename = maybe_download('text8.zip', 31344016)


# Read the data into a string.

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)  # return a string
print('Data size %d' % len(words))

# Build the dictionary and replace rare words with UNK token.

vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]  # UNK: unknown word in Tensorflow; 由于 'UNK'对应出现次数后面要赋值，故这里用list，不用tuple.
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))  # 返回words中，top 49999多的words及其出现次数
    dictionary = dict()
    for word, _ in count:  # 词按出现频率由高到低的顺序，依次iterate
        dictionary[word] = len(dictionary)  # dictioanry 中 value说明了word 频率的名次, dictionary 中, word都在 .key中，是按words统计时的顺序排列
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]  # word同时出现在 words 和 dictioanry中（即word在出现频率为 Top 49999的词的集合中）
        else:
            index = 0  # dictionary['UNK'] # 出现频率较低的words，index = 0，都统计为 UNK
            unk_count = unk_count + 1
        data.append(index)  # data记录了原文中（words）,依次出现的词的出现频率的名次
    count[0][1] = unk_count        # count[0] = ['UNK', 418391], 即 count[0]记录了 UNK的个数，count[i] 记录了 第i多的word的个数
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

# data: 原文中，依次出现的words的出现次数；出现次数按高到低排名超过49999的，word都记为UNK，排名为0
# count: 每个元素格式为 ('word', #出现次数)，元素按出现次数从大到小排列('UNK'无论出现次数多少都排在第一，即：count[0] = ['UNK',418391], count[1] = ('the',1061396))
# dictionary: key: word, value: 出现次数的排名（是对count的一个处理）
# reverse_dictionary: key: 出现次数的排名； key: word. (key按数字由小到大排列，是对dictionary的一个处理）


data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.


# Function to generate a training batch for the skip-gram model.
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):  # batch_size: the size of a batch of sample data.
  global data_index
  assert batch_size % num_skips == 0    # batch_size中应当有整数个skips
  assert num_skips <= 2 * skip_window   # 这个条件不满足，则一个skips中的位置不能被两个skip_window中的元素填满
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)   #随机生成的一个array, shape为(batch_size,)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span) #buffer是一个bounded 双向队列，长度上限是span
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)  #求余数
  # buffer 一次读进数量为 span的words.
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of buffer (length of a span). 因为buffer的长度是始终是 2*skip_window + 1
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1) # target 在 一个span的范围内[0, span-1]，除了已经选择过的，在剩余的words中随机选择一个值作为新的target index
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])   # batch中填入了 num_skips的words之后，buffer中就加入一个新的word，同时将最左边的元素弹出
    data_index = (data_index + 1) % len(data)
  return batch, labels


print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size)) #shape: (16,)，从 [0,99]中不放回随机取16个，之后作为index集合
num_sampled = 64 # Number of negative examples to sample.


graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])  #shape = (128,)
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])  # shape = (128,1)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)  #shape = (16,)

    # Variables.
    embeddings = tf.Variable(  # shape = (50000,128)
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))  # 词典中words的数量为vocabulary_size，每个词用长度为 embedding_size的向量来表示。
        # 随机得到的张量的shape由第一个参数指定，每个元素由[-1,1)之间均匀分布得到
    softmax_weights = tf.Variable(   # shape = (50000,128)
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size])) #shape = (50000,)
    # embed 通过一个 softmax 分类器，这是一个线性函数 WX+b，W,b由这里的softmax_weights, softmax_biases给出
    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)   # shape: (128, 128)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:

    # embeddings shape: (50000, 128), 50000 words in vocabulary, 128 features.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))    # norm shape: (50000, 1), 每个行向量都是一个 embedding向量元素的平方和。
    normalized_embeddings = embeddings / norm   # shape为(50000,128)和 (50000,1)的两个向量的division，结果shape = (50000,128) 是embeddings的每个行向量除上norm的每个行向量。
    valid_embeddings = tf.nn.embedding_lookup(  # shape = (16,128)
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings)) # shape = (16, 50000)


num_steps = 100001


with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)  # 运行 optimizer()
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]  # 出现次数的名次为valid_example[i]的word
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1] # 返回的列表包含与valid_word相似度最高的top_k个words的相似度按从高到低在sim[i]中的indices.
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):  # 按相似度从高到低依次输出words.
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()   # shape (50000,128)

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact') # 定义一个 tsne实例
# 对数据 final_embeddings中 1-400 这400个词的embedding向量进行降维
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])   # shape (400, 2)


def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]  # x: 第i个word, y: 对应的降维后的向量
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()


words = [reverse_dictionary[i] for i in range(1, num_points+1)] # 提取前400名的words
plot(two_d_embeddings, words)



