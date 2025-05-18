'''
MNIST Static HyperNetwork Example (Simplified Version)
Based on the paper "Hypernetworks" by David Ha, Andrew Dai, and Quoc V. Le.

This script demonstrates static hypernetworks on MNIST dataset,
comparing a standard CNN to a hypernetwork-based CNN.
'''

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Import Keras for MNIST data loading
from keras.datasets import mnist

# Configure numpy output format
np.set_printoptions(precision=5, edgeitems=8, linewidth=200)

# Utility Functions
def orthogonal(shape):
    """Generate orthogonal matrix for weight initialization"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def orthogonal_initializer(scale=1.0):
    """TensorFlow initializer using orthogonal matrices"""
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape) * scale, dtype)
    return _initializer

def super_linear(x, output_size, scope=None, reuse=False, init_w="ortho", 
                weight_start=0.0, use_bias=True, bias_start=0.0):
    """Fully connected layer with customizable initialization"""
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or "linear"):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        w_init = None
        x_size = shape[1]
        h_size = output_size
        
        if init_w == "zeros":
            w_init = tf.constant_initializer(0.0)
        elif init_w == "constant":
            w_init = tf.constant_initializer(weight_start)
        elif init_w == "gaussian":
            w_init = tf.random_normal_initializer(stddev=weight_start)
        elif init_w == "ortho":
            w_init = orthogonal_initializer(1.0)

        w = tf.get_variable("super_linear_w",
            [shape[1], output_size], tf.float32, initializer=w_init)
        if use_bias:
            b = tf.get_variable("super_linear_b", [output_size], tf.float32,
                initializer=tf.constant_initializer(bias_start))
            return tf.matmul(x, w) + b
        return tf.matmul(x, w)

# Dataset class for MNIST
class DataSet(object):
    def __init__(self, images, labels, augment=False):
        # Convert to float32 and reshape
        images = images.astype(np.float32)
        self.image_size = 28
        self._num_examples = len(images)
        images = np.reshape(images, (self._num_examples, self.image_size, self.image_size, 1))
        
        # Shuffle data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = images[perm]
        self._labels = labels[perm]
        
        # Configuration
        self._augment = augment
        self.pointer = 0
        self.upsize = 1 if self._augment else 0
        self.min_upsize = 2
        self.max_upsize = 2
        self.random_perm_mode = False
        self.num_classes = 10

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size=100, with_label=True, one_hot=False):
        """Get next batch of data with optional augmentation"""
        # Update pointer
        if self.pointer >= self.num_examples - 2*batch_size:
            self.pointer = 0
        else:
            self.pointer += batch_size
            
        # Set up augmentation
        result = []
        upsize_amount = np.random.randint(self.upsize*self.min_upsize, self.upsize*self.max_upsize+1)
        
        # Helper functions for augmentation
        def upsize_row_once(img):
            old_size = img.shape[0]
            new_size = old_size + 1
            new_img = np.zeros((new_size, img.shape[1], 1))
            rand_row = np.random.randint(1, old_size-1)
            new_img[0:rand_row,:] = img[0:rand_row,:]
            new_img[rand_row+1:,:] = img[rand_row:,:]
            new_img[rand_row,:] = 0.5 * (new_img[rand_row-1,:] + new_img[rand_row+1,:])
            return new_img
        
        def upsize_col_once(img):
            old_size = img.shape[1]
            new_size = old_size + 1
            new_img = np.zeros((img.shape[0], new_size, 1))
            rand_col = np.random.randint(1, old_size-1)
            new_img[:,0:rand_col,:] = img[:,0:rand_col,:]
            new_img[:,rand_col+1:,:] = img[:,rand_col:,:]
            new_img[:,rand_col,:] = 0.5 * (new_img[:,rand_col-1,:] + new_img[:,rand_col+1,:])
            return new_img
        
        def upsize_me(img, n=2):
            new_img = img
            for i in range(n):
                new_img = upsize_row_once(new_img)
                new_img = upsize_col_once(new_img)
            return new_img

        # Create augmented batch
        for data in self._images[self.pointer:self.pointer+batch_size]:
            result.append(self.distort_image(upsize_me(data, upsize_amount), upsize_amount))
            
        # Validate batch size
        if len(result) != batch_size:
            print("Error: batch size mismatch, pointer =", self.pointer)
        assert(len(result) == batch_size)
        
        # Get corresponding labels
        result_labels = self._labels[self.pointer:self.pointer+batch_size]
        assert(len(result_labels) == batch_size)
        
        # Convert to one-hot if needed
        if one_hot:
            result_labels = np.eye(self.num_classes)[result_labels]
            
        # Return batch with or without labels
        if with_label:
            return self.scramble_batch(np.array(result, dtype=np.float32)), result_labels
        return self.scramble_batch(np.array(result, dtype=np.float32))

    def scramble_batch(self, batch):
        """Apply random permutation to batch if enabled"""
        if self.random_perm_mode:
            batch_size = len(batch)
            result = np.copy(batch)
            result = result.reshape(batch_size, self.image_size*self.image_size)
            result = result[:, self.random_key]
            return result
        else:
            return batch
    
    def distort_image(self, img, upsize_amount):
        """Apply random distortion to image"""
        row_distort = np.random.randint(0, self.image_size+upsize_amount-self.image_size+1)
        col_distort = np.random.randint(0, self.image_size+upsize_amount-self.image_size+1)
        result = np.zeros(shape=(self.image_size, self.image_size, 1), dtype=np.float32)
        result[:, :, :] = img[row_distort:row_distort+self.image_size, 
                             col_distort:col_distort+self.image_size, :]
        return result

    def shuffle_data(self):
        """Shuffle the dataset"""
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

# MNIST model class
class MNIST(object):
    def __init__(self, hps_model, reuse=False, gpu_mode=True, is_training=True):
        self.is_training = is_training
        with tf.variable_scope('conv_mnist', reuse=reuse):
            if not gpu_mode:
                with tf.device("/cpu:0"):
                    print("Model using CPU")
                    self.build_model(hps_model)
            else:
                self.build_model(hps_model)

    def build_model(self, hps_model):
        """Build the MNIST model with optional hypernetwork"""
        self.hps = hps_model
        self.model_path = self.hps.model_path
        self.model_save_path = self.model_path + 'mnist'
        
        # Placeholders for input data and labels
        self.batch_images = tf.placeholder(tf.float32, 
                                         [self.hps.batch_size, self.hps.x_dim, self.hps.x_dim, self.hps.c_dim])
        self.batch_labels = tf.placeholder(tf.float32, 
                                         [self.hps.batch_size, self.hps.num_classes])
        
        # Architecture settings
        f_size = 7
        in_size = 16
        out_size = 16
        z_dim = 4

        # First convolutional layer (same for both standard and hyper mode)
        conv1_weights = tf.Variable(tf.truncated_normal([f_size, f_size, 1, out_size], stddev=0.01), 
                                   name="conv1_weights")
        
        # Second convolutional layer - depends on hyper_mode
        if self.hps.hyper_mode:
            # Hypernetwork implementation - generating weights
            w1 = tf.get_variable('w1', [z_dim, out_size*f_size*f_size], 
                              initializer=tf.truncated_normal_initializer(stddev=0.01))
            b1 = tf.get_variable('b1', [out_size*f_size*f_size], 
                              initializer=tf.constant_initializer(0.0))
            z2 = tf.get_variable("z_signal_2", [1, z_dim], tf.float32, 
                              initializer=tf.truncated_normal_initializer(0.01))
            w2 = tf.get_variable('w2', [z_dim, in_size*z_dim], 
                              initializer=tf.truncated_normal_initializer(stddev=0.01))
            b2 = tf.get_variable('b2', [in_size*z_dim], 
                              initializer=tf.constant_initializer(0.0))
            
            # Generate weights using hypernetwork
            h_in = tf.matmul(z2, w2) + b2
            h_in = tf.reshape(h_in, [in_size, z_dim])
            h_final = tf.matmul(h_in, w1) + b1
            kernel2 = tf.reshape(h_final, (out_size, in_size, f_size, f_size))
            conv2_weights = tf.transpose(kernel2)
        else:
            # Standard convolutional weights
            conv2_weights = tf.Variable(tf.truncated_normal([f_size, f_size, in_size, out_size], stddev=0.01), 
                                      name="conv2_weights")

        self.conv1_weights = conv1_weights
        self.conv2_weights = conv2_weights

        # Build the network
        conv1_biases = tf.Variable(tf.zeros([in_size]), name="conv1_biases")
        net = tf.nn.conv2d(self.batch_images, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(net + conv1_biases)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2_biases = tf.Variable(tf.zeros([out_size]), name="conv2_biases")
        net = tf.nn.conv2d(net, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(net + conv2_biases)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        # Flatten and create output layer
        net = tf.reshape(net, [self.hps.batch_size, -1])
        net = super_linear(net, self.hps.num_classes, scope='fc_final')
        
        # Outputs
        self.logits = net
        self.probabilities = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, 1)

        # Loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.batch_labels)
        self.loss = tf.reduce_mean(cross_entropy)
        
        # Optimizer
        self.lr = tf.Variable(self.hps.lr, trainable=False)
        optimizer = tf.train.AdamOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.hps.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Model saver
        self.saver = tf.train.Saver(tf.global_variables())
        
    def update_lr(self, sess):
        """Update learning rate with decay"""
        lr = sess.run(self.lr)
        lr *= self.hps.lr_decay
        sess.run(tf.assign(self.lr, np.maximum(lr, self.hps.min_lr)))
        
    def partial_train(self, sess, batch_images, batch_labels):
        """Run one training step"""
        _, loss, pred, lr = sess.run(
            (self.train_op, self.loss, self.predictions, self.lr),
            feed_dict={self.batch_images: batch_images, self.batch_labels: batch_labels}
        )
        return loss, pred, lr
    
    def partial_eval(self, sess, batch_images, batch_labels):
        """Run one evaluation step"""
        loss, pred = sess.run(
            (self.loss, self.predictions),
            feed_dict={self.batch_images: batch_images, self.batch_labels: batch_labels}
        )
        return loss, pred

def load_mnist_data():
    """Load MNIST data using Keras"""
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Split test into validation and test
    val_size = 10000
    x_val = x_test[:val_size]
    y_val = y_test[:val_size]
    x_test = x_test[val_size:]
    y_test = y_test[val_size:]
    
    # Create a struct-like object to match the original API
    class MNISTData:
        class DataSet:
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels
        
        def __init__(self, train, validation, test):
            self.train = train
            self.validation = validation
            self.test = test
    
    train_dataset = MNISTData.DataSet(x_train, y_train)
    validation_dataset = MNISTData.DataSet(x_val, y_val)
    test_dataset = MNISTData.DataSet(x_test, y_test)
    
    return MNISTData(train_dataset, validation_dataset, test_dataset)

def read_data_sets(mnist_data):
    """Create DataSet objects from MNIST data"""
    class DataSets(object):
        pass
    data_sets = DataSets()

    data_sets.train = DataSet(mnist_data.train.images, mnist_data.train.labels, augment=True)
    data_sets.valid = DataSet(mnist_data.validation.images, mnist_data.validation.labels, augment=False)
    data_sets.test = DataSet(mnist_data.test.images, mnist_data.test.labels, augment=False)
    
    return data_sets

def process_epoch(sess, model, dataset, train_mode=False, print_every=0):
    """Process one epoch of data"""
    num_examples = dataset.num_examples
    batch_size = model.hps.batch_size
    total_batch = int(num_examples / batch_size)
    
    avg_loss = 0.
    avg_pred_error = 0.
    lr = model.hps.lr

    for i in range(total_batch):
        # Get batch and convert labels to one-hot
        batch_images, batch_labels = dataset.next_batch(batch_size, with_label=True, one_hot=False)
        batch_labels_onehot = np.eye(dataset.num_classes)[batch_labels]

        # Train or evaluate
        if train_mode:
            loss, pred, lr = model.partial_train(sess, batch_images, batch_labels_onehot)
            model.update_lr(sess)
        else:
            loss, pred = model.partial_eval(sess, batch_images, batch_labels_onehot)

        # Calculate prediction error
        pred_error = 1.0 - np.sum((pred == batch_labels)) / float(batch_size)
        
        # Print progress
        if print_every > 0 and i > 0 and i % print_every == 0:
            print("Batch:", '%d' % (i), "/", '%d' % (total_batch),
                  "loss=", "{:.4f}".format(loss),
                  "err=", "{:.4f}".format(pred_error))
                
        # Check for NaN/Inf
        assert(loss < 1000000)

        # Update averages
        avg_loss += loss / num_examples * batch_size
        avg_pred_error += pred_error / num_examples * batch_size
    
    return avg_loss, avg_pred_error, lr

def train_model(sess, model, eval_model, mnist, num_epochs, save_model=False):
    """Train the model for specified number of epochs"""
    best_valid_loss = 100.
    best_valid_pred_error = 1.0
    eval_loss = 100.
    eval_pred_error = 1.0
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_pred_error, lr = process_epoch(
            sess, model, mnist.train, train_mode=True, print_every=10
        )
        
        # Validation phase
        valid_loss, valid_pred_error, _ = process_epoch(
            sess, eval_model, mnist.valid, train_mode=False
        )
        
        # Check if this is the best model so far
        if valid_pred_error <= best_valid_pred_error:
            best_valid_pred_error = valid_pred_error
            best_valid_loss = valid_loss
            
            # Evaluate on test set
            eval_loss, eval_pred_error, _ = process_epoch(
                sess, eval_model, mnist.test, train_mode=False
            )
            
            # Save model if requested
            if save_model:
                model.save_model(sess, epoch)

        # Print epoch results
        print("Epoch:", '%d' % (epoch),
              "train_loss=", "{:.4f}".format(train_loss),
              "train_err=", "{:.4f}".format(train_pred_error),
              "valid_err=", "{:.4f}".format(valid_pred_error),
              "best_valid_err=", "{:.4f}".format(best_valid_pred_error),
              "test_err=", "{:.4f}".format(eval_pred_error),
              "lr=", "{:.6f}".format(lr))

def show_filter_stats(conv_filter):
    """Display statistics for a convolutional filter"""
    print("Filter shape:", conv_filter.shape)
    print("mean =", np.mean(conv_filter))
    print("stddev =", np.std(conv_filter))
    print("max =", np.max(conv_filter))
    print("min =", np.min(conv_filter))
    print("median =", np.median(conv_filter))

def count_parameters(session):
    """Count and display trainable parameters"""
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        print(var.name, var.get_shape(), num_param)
    print("Total trainable variables =", count_t_vars)
    return count_t_vars

def main():
    """Main function to run the experiment"""
    print("Starting MNIST Hypernetwork Example...")
    
    # Hyperparameters
    class HParams(object):
        pass

    hps_model = HParams()
    hps_model.lr = 0.005
    hps_model.lr_decay = 0.999
    hps_model.min_lr = 0.0001
    hps_model.is_training = True
    hps_model.x_dim = 28
    hps_model.num_classes = 10
    hps_model.c_dim = 1
    hps_model.batch_size = 1000  # Match the original notebook batch size
    hps_model.grad_clip = 100.0
    hps_model.hyper_mode = False
    hps_model.model_path = '/tmp/'
    
    # Load MNIST data
    mnist_data = load_mnist_data()
    mnist = read_data_sets(mnist_data)
    
    # Train standard CNN model
    print("\nTraining standard CNN model...")
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST(hps_model)
        sess.run(tf.global_variables_initializer())
        
        # Train for specified number of epochs
        num_epochs = 50  # Match the original notebook epoch count
        train_model(sess, model, model, mnist, num_epochs, save_model=False)  # Train for 50 epochs like original notebook
        
        # Display filter statistics
        conv_filter = sess.run(model.conv2_weights)
        show_filter_stats(conv_filter)
        
        # Count parameters
        count_parameters(sess)
    
    # Train hypernetwork CNN model
    print("\nTraining hypernetwork CNN model...")
    hps_model.hyper_mode = True
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = MNIST(hps_model)
        sess.run(tf.global_variables_initializer())
        
        # Train for specified number of epochs (50 epochs like original notebook)
        train_model(sess, model, model, mnist, num_epochs, save_model=False)
        
        # Display filter statistics
        conv_filter = sess.run(model.conv2_weights)
        show_filter_stats(conv_filter)
        
        # Count parameters
        count_parameters(sess)
    
    print("\nExecution complete.")

if __name__ == "__main__":
    main()
