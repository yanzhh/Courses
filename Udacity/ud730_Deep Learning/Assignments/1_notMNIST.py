# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:01:17 2018

@author: Arc
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)



##########################################################################################

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(data_root)
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)


##########################################################################################
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)      #得到folder中所有文件名构成的列表
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)        #dataset[i].shape = 28 x 28, 用来存储每张图的pixels
  print(folder)
  num_images = 0
  for image in image_files:          #在所有的images文件中 iteration
    image_file = os.path.join(folder, image)   #得到image的绝对路径
    try:
      image_data = (imageio.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth           # imageio.imread(): Returns a numpy array, which comes with a dict of meta data at its ‘meta’ attribute. 
      #imageio.imread(image_file): 28x28, 得到的是每个像素的灰度值？， (level - 128)/128：将level归一化
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:        #将出错信息存储在变量e中
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:       # open(filename,'wb'): 以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。
            #创建或修改filename存储在变量 set_filename 中的二进制文件，并将其存储在变量f中。这里即.pickle文件。
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


##################################################################################

#Problem 1

folder = train_folders[0]
file = os.listdir(train_folders[0])[0]
path = os.path.join(folder, file)
display(Image( filename = path))

#################################################################################

#Problem 2

sample_set = train_datasets[0]
with open(sample_set, 'rb') as f:
    sample_dataset = pickle.load(f)
    sample_data =sample_dataset[0]
    sample_image = sample_data*pixel_depth + pixel_depth/2

plt.imshow(sample_image, cmap = 'gray')
plt.show
##################################################################################
# Problem 3
# Another check: we expect the data to be balanced across classes. Verify that.
# 验证每个class中的文件数量大致相当
set_sizes= []
stds = []

for train_file in train_datasets:
    with open(sample_set, 'rb') as f:
        dataset = pickle.load(f)
        set_sizes.append(dataset.shape[0])
    
print('sizes of each training set', set_sizes)    


####################################################################################

# Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune train_size as needed. The labels will be stored into a separate array of integers 0 through 9.

# Also create a validation dataset for hyperparameter tuning.

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size) #生成shape为 valid_size x image_size x image_size 的 ndarray, 以及对应的labels array.
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes        #每组有多少验证数据
  tsize_per_class = train_size // num_classes        #每组有多少训练数据
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):    #lable: 用数字表示对应字母    
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000          #
valid_size = 10000           #验证数据数量
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size) #测试集中不需验证数据，因此valid_size = 0)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

###################################################################################
#Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)  #按列将数据打散了。
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


#####################################################################################
# Problem 4
# Convince yourself that the data is still good after shuffling!
set_sizes= []
stds = []

for train_file in train_datasets:
    with open(sample_set, 'rb') as f:
        dataset = pickle.load(f)
        set_sizes.append(dataset.shape[0])
    
print('sizes of each training set', set_sizes)    
##################################################################################
#Finally, let's save the data for later reuse:

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
  statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

#####################################################################################
#Problem 5
# By construction, this dataset might contain a lot of overlapping samples, 
# including training data that's also contained in the validation and test set! 
# Overlap between training and test can skew the results if you expect to use your model 
# in an environment where there is never an overlap, but are actually ok 
# if you expect to see training samples recur when you use it. 
# Measure how much overlap there is between training, validation and test samples.

#Optional questions:

#What about near duplicates between datasets? (images that are almost identical)
#Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.

print('Hash')

train_dataset_set = set([hash(str(x)) for x in train_dataset])
valid_dataset_set = set([hash(str(x)) for x in valid_dataset])
test_dataset_set = set([hash(str(x)) for x in test_dataset])

print('train data set: ' + str(len(train_dataset)) + ' set: ' + str(len(train_dataset_set)))
print('valid data set: ' + str(len(valid_dataset)) + ' set: ' + str(len(valid_dataset_set)))
print('test data set: ' + str(len(test_dataset)) + ' set: ' + str(len(test_dataset_set)))

overlap_train_valid = train_dataset_set & valid_dataset_set
overlap_train_test = train_dataset_set & test_dataset_set
overlap_valid_test = test_dataset_set & valid_dataset_set

print('overlap_train_valid: ' + str(len(overlap_train_valid)))
print('overlap_train_test: ' + str(len(overlap_train_test)))
print('overlap_valid_test: ' + str(len(overlap_valid_test)))

print('MD5')
from hashlib import md5
train_dataset_set_2 = set([ md5(x).hexdigest() for x in train_dataset])
valid_dataset_set_2 = set([ md5(x).hexdigest() for x in valid_dataset])
test_dataset_set_2 = set([ md5(x).hexdigest() for x in test_dataset])

print('train data set: ' + str(len(train_dataset)) + ' set: ' + str(len(train_dataset_set_2)))
print('valid data set: ' + str(len(valid_dataset)) + ' set: ' + str(len(valid_dataset_set_2)))
print('test data set: ' + str(len(test_dataset)) + ' set: ' + str(len(test_dataset_set_2)))

overlap_train_valid_2 = train_dataset_set_2 & valid_dataset_set_2
overlap_train_test_2 = train_dataset_set_2 & test_dataset_set_2
overlap_valid_test_2 = test_dataset_set_2 & valid_dataset_set_2

print('overlap_train_valid: ' + str(len(overlap_train_valid_2)))
print('overlap_train_test: ' + str(len(overlap_train_test_2)))
print('overlap_valid_test: ' + str(len(overlap_valid_test_2)))



##########################################################################
f = open(pickle_file,'rb')
save_files = pickle.load(f)
f.close()
train_dataset = save_files['train_dataset']
train_labels = save_files['train_labels']
valid_dataset = save_files['valid_dataset']
valid_labels = save_files['valid_labels']
test_dataset = save_files['test_dataset']
test_labels = save_files['test_labels']

######################################################################################
# Problem 6
# Let's get an idea of what an off-the-shelf classifier can give you on this data. 
# It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.

# Train a simple model on this data using 50, 100, 1000 and 5000 training samples. 
# Hint: you can use the LogisticRegression model from sklearn.linear_model.

# Optional question: train an off-the-shelf model on all the data!

LR = LogisticRegression(solver = 'lbfgs')
LR50 = LogisticRegression(solver = 'lbfgs')
LR100 = LogisticRegression(solver = 'lbfgs')
LR1000 = LogisticRegression(solver = 'lbfgs')
LR5000 = LogisticRegression(solver = 'lbfgs')



sizes = [50, 100, 1000, 5000]
train_dataset50_flat = np.asarray([x.flatten() for x in train_dataset[:sizes[0]]])
train_dataset100_flat = np.asarray([x.flatten() for x in train_dataset[:sizes[1]]])
train_dataset1000_flat = np.asarray([x.flatten() for x in train_dataset[:sizes[2]]])
train_dataset5000_flat = np.asarray([x.flatten() for x in train_dataset[:sizes[3]]])
model50 = LR50.fit(train_dataset50_flat, train_labels[:sizes[0]])
model100 = LR100.fit(train_dataset100_flat, train_labels[:sizes[1]])
model1000 = LR1000.fit(train_dataset1000_flat, train_labels[:sizes[2]])
model5000 = LR5000.fit(train_dataset5000_flat, train_labels[:sizes[3]])
test_dataset_flat = np.asarray([x.flatten() for x in test_dataset])
score50 = model50.score(test_dataset_flat, test_labels)
score100 = model50.score(test_dataset_flat, test_labels)
score1000 = model1000.score(test_dataset_flat, test_labels)
score5000 = model5000.score(test_dataset_flat, test_labels)
print('score50',score50,'score100', score100, 'score1000', score1000, 'score5000', score5000)

train_dataset_flat = np.asarray([x.flatten() for x in train_dataset])

LR.fit(train_dataset_flat, train_labels)
score = LR.score(test_dataset_flat, test_labels)
print('score_total', score)

