import tensorflow as tf
from tensorflow.contrib import rnn
from components import processors
from instances import layer_instances
from operator import mul
import numpy as np

class Single_layer_interface(processors.BaseFeedforwardProcessor):
    """
    A processor with a single fully-connected layer. Can be used as an interface (for projection, reshaping, etc.)
    """

    def initialize_vars(self, output_size, activation_type='relu', has_biases=True, trainable=True):
        """
        Inputs:

        output_size : 		(int) number of output units
        activation_type : 	(str) the activation type of the FC layer
        has_biases : 		(bool) whether or not to use bias
        trainable : 		(bool) if False, the layer weights and biases are not added to the list of trainable variables.
        """
        self.trainable = trainable
        layer1 = layer_instances.FCLayer(name=self.name + '/l1', input_size=self.output_size, batch_size=self.batch_size,
                                trainable=self.trainable)
        layer1.initialize_vars(output_channels=reduce(mul, output_size), activation_type=activation_type,
                               has_biases=has_biases)
        self.add_layer(layer1)
        self.output_size = output_size

    def run(self, X):
        output = super(Single_layer_interface, self).run(X)

        return tf.reshape(output, [self.batch_size] + self.output_size)


class FC_k_layer(processors.BaseFeedforwardProcessor):
    """
    A multi-layered perceptron
    """

    def initialize_vars(self, num_layers, num_features, output_size, activation_type='relu'):
        """
        Inputs:

        num_layers : 		(int) number of FC layers
        num_features : 		(int) number of units in the intermediate layers
        output_size : 		(int) number of units in the output layer
        activation_type : 	(str) the activation type of the final FC layer

        """
        layers_list = []
        last_output_size = self.output_size
        for ii in range(num_layers):
            if ii == num_layers - 1:
                # construct last layer
                layers_list.append(layer_instances.FCLayer(name=self.name + '/l' + str(ii + 1), input_size=last_output_size,
                                                  batch_size=self.batch_size))
                layers_list[-1].initialize_vars(output_channels=reduce(mul, output_size),
                                                activation_type=activation_type)
            else:
                # construct nonterminal layers
                layers_list.append(layer_instances.FCLayer(name=self.name + '/l' + str(ii + 1), input_size=last_output_size,
                                                  batch_size=self.batch_size))
                layers_list[-1].initialize_vars(output_channels=num_features)
                last_output_size = [1, 1, num_features]

        for ii in range(num_layers):
            self.add_layer(layers_list[ii])

        self.output_size = output_size

    def run(self, X):

        output = super(FC_k_layer, self).run(X)
        return tf.reshape(output, [self.batch_size] + self.output_size)


class PSVRT_cnn(processors.BaseFeedforwardProcessor):
    """
    A multilayer simese convolutional net. Input channels are processed separately using shared conv weights.
    """

    def initialize_vars(self, num_categories,
                        num_CP_layers, num_CP_features, num_FC_layers, num_FC_features,
                        initial_conv_rf_size, interm_conv_rf_size, pool_rf_size=[3, 3], stride_size=[2, 2],
                        activation_type='relu', trainable=True, hamstring_factor=1.0,
                        global_pool=False):
        """
        Inputs:

        num_CP_layers :     (int) number of conv-pool layer pairs (e.g. if 2, it means there are of total 4 layers)
        num_features : 		(int) number of convolution filters per layer
        conv_rf_size : 		(list) The receptive field size of the convolution kernel,  [height, width]
        pool_rf_size :		(list) The receptive field size of the pool kernel,  [height, width]
        stride_size : 		(list) The pooling stride side, [height, width]
        activation_type : 	(str) the activation type of the conv layers
        attn :				(bool) whether or not to use the initial spatial attention layer
        global_pool :  		(bool) whether or not to use the final global pooling layer
        trainable :  		(bool) whether or not to include its parameters to the list of trainable variables
        """
        self.global_pool = global_pool
        layer_list = []

        intermediate_output_size = self.get_output_size()
        layer_ind = 0

        for ii in range(num_CP_layers):
            layer_ind += 1
            conv_rf_size = initial_conv_rf_size if (ii == 0) else interm_conv_rf_size
            num_features = int(num_CP_features*hamstring_factor) if (ii==0) else int(num_features*interm_conv_rf_size[0])

            # construct conv layer
            layer_list.append(
                layer_instances.Conv2dLayer(name=self.name + '/conv_' + str(ii + 1), input_size=intermediate_output_size,
                                            batch_size=self.batch_size, trainable=trainable))
            layer_list[-1].initialize_vars(rf_size=conv_rf_size, output_channels=num_features, stride=[1, 1],
                                           activation_type=activation_type)
            intermediate_output_size = layer_list[-1].get_output_size()
            self.add_layer(layer_list[-1])

            # construct pool layer
            layer_list.append(
                layer_instances.Maxpool2dLayer(name=self.name + '/pool_' + str(ii + 1), input_size=intermediate_output_size,
                                               batch_size=self.batch_size))
            layer_list[-1].initialize_vars(rf_size=pool_rf_size, stride=stride_size)
            intermediate_output_size = layer_list[-1].get_output_size()
            self.add_layer(layer_list[-1])

            # construct global pool layer
            if ii == num_CP_layers-1:
                if self.global_pool:
                    pool_size = intermediate_output_size[:2]
                else:
                    pool_size = [2,2]
                layer_list.append(
                    layer_instances.Maxpool2dLayer(name=self.name + '/global_pool',
                                                   input_size=intermediate_output_size,
                                                   batch_size=self.batch_size))
                layer_list[-1].initialize_vars(rf_size=pool_size, stride=[1,1])
                intermediate_output_size = layer_list[-1].get_output_size()

                self.add_layer(layer_list[-1])

        for jj in range(num_FC_layers+1):
            layer_ind += 1
            num_features = int(num_FC_features*hamstring_factor) if (jj<num_FC_layers-1) else num_categories
            activation = activation_type if (jj<num_FC_layers-1) else 'raw'

            layer_list.append(layer_instances.FCLayer(name=self.name + '/FC_' + str(jj + 1), input_size=intermediate_output_size,
                                              batch_size=self.batch_size))
            layer_list[-1].initialize_vars(output_channels=num_features, activation_type=activation)
            intermediate_output_size = layer_list[-1].get_output_size()

            self.add_layer(layer_list[-1])

            if jj == 0: # add dropout layer
                layer_list.append(layer_instances.DropoutLayer(name=self.name + '/dropout_' + str(jj + 1),
                                                                input_size=intermediate_output_size,
                                                                batch_size=self.batch_size))
                layer_list[-1].initialize_vars(dropout_multiplier=1.)
                intermediate_output_size = layer_list[-1].get_output_size()

                self.add_layer(layer_list[-1])

        self.output_size = layer_list[-1].get_output_size()


    def run(self, X, dropout_keep_prob=1.):
        intermediate = X
        for current_layer in self.layer_list:
            if isinstance(current_layer, layer_instances.DropoutLayer):
                intermediate = current_layer.run(intermediate, dropout_keep_prob=dropout_keep_prob)
            else:
                intermediate = current_layer.run(intermediate)

        return intermediate

class PSVRT_siamesenet(processors.BaseFeedforwardProcessor):
    """
    A multilayer simese convolutional net. Input channels are processed separately using shared conv weights.
    """

    def initialize_vars(self, num_categories,
                        num_items, organization,
                        num_CP_layers, num_CP_features, num_FC_layers, num_FC_features,
                        initial_conv_rf_size, interm_conv_rf_size, pool_rf_size=[3, 3], stride_size=[2, 2],
                        activation_type='relu', trainable=True, hamstring_factor=1.0,
                        global_pool=False):
        """
        Inputs:

        num_CP_layers :     (int) number of conv-pool layer pairs (e.g. if 2, it means there are of total 4 layers)
        num_features : 		(int) number of convolution filters per layer
        conv_rf_size : 		(list) The receptive field size of the convolution kernel,  [height, width]
        pool_rf_size :		(list) The receptive field size of the pool kernel,  [height, width]
        stride_size : 		(list) The pooling stride side, [height, width]
        activation_type : 	(str) the activation type of the conv layers
        attn :				(bool) whether or not to use the initial spatial attention layer
        global_pool :  		(bool) whether or not to use the final global pooling layer
        trainable :  		(bool) whether or not to include its parameters to the list of trainable variables
        """
        self.global_pool = global_pool
        self.organization = organization
        self.num_items = num_items
        if (self.organization == 'obj') | (self.organization=='full'):
            self.input_size[2]=self.input_size[2]*self.num_items

        layer_list = []
        self.output_size[2] = 1 #Channels are processed separately via regular_n_siamese convnet

        intermediate_output_size = self.get_output_size()
        layer_ind = 0

        for ii in range(num_CP_layers):
            layer_ind += 1
            conv_rf_size = initial_conv_rf_size if (ii == 0) else interm_conv_rf_size
            num_features = int(num_CP_features*hamstring_factor) if (ii==0) else int(num_features*interm_conv_rf_size[0])

            # construct conv layer
            layer_list.append(
                layer_instances.Conv2dLayer(name=self.name + '/conv_' + str(ii + 1), input_size=intermediate_output_size,
                                            batch_size=self.batch_size, trainable=trainable))
            layer_list[-1].initialize_vars(rf_size=conv_rf_size, output_channels=num_features, stride=[1, 1],
                                           activation_type=activation_type)
            intermediate_output_size = layer_list[-1].get_output_size()
            self.add_layer(layer_list[-1])

            # construct pool layer
            layer_list.append(
                layer_instances.Maxpool2dLayer(name=self.name + '/pool_' + str(ii + 1), input_size=intermediate_output_size,
                                      batch_size=self.batch_size))
            layer_list[-1].initialize_vars(rf_size=pool_rf_size, stride=stride_size)
            intermediate_output_size = layer_list[-1].get_output_size()
            self.add_layer(layer_list[-1])

            # construct global pool layer
            if ii == num_CP_layers-1:
                if self.global_pool:
                    pool_size = intermediate_output_size[:2]
                else:
                    pool_size = [2,2]
                layer_list.append(
                    layer_instances.Maxpool2dLayer(name=self.name + '/global_pool',
                                                   input_size=intermediate_output_size,
                                                   batch_size=self.batch_size))
                layer_list[-1].initialize_vars(rf_size=pool_size, stride=[1,1])
                intermediate_output_size = layer_list[-1].get_output_size()

                self.add_layer(layer_list[-1])

        self.num_siamese_layers = len(layer_list)

        # Channels outputted by simese net are combined
        if (self.organization == 'obj') | (self.organization=='full'):
            intermediate_output_size[2] = intermediate_output_size[2]*self.num_items
            self.output_size[2] = self.output_size[2]*self.num_items
        for jj in range(num_FC_layers+1):
            layer_ind += 1
            num_features = int(num_FC_features*hamstring_factor) if (jj<num_FC_layers-1) else num_categories
            activation = activation_type if (jj<num_FC_layers-1) else 'raw'

            layer_list.append(layer_instances.FCLayer(name=self.name + '/FC_' + str(jj + 1), input_size=intermediate_output_size,
                                              batch_size=self.batch_size))
            layer_list[-1].initialize_vars(output_channels=num_features, activation_type=activation)
            intermediate_output_size = layer_list[-1].get_output_size()

            self.add_layer(layer_list[-1])

            if jj == 0: # add dropout layer
                layer_list.append(layer_instances.DropoutLayer(name=self.name + '/dropout_' + str(jj + 1),
                                                                input_size=intermediate_output_size,
                                                                batch_size=self.batch_size))
                layer_list[-1].initialize_vars(dropout_multiplier=1.)
                intermediate_output_size = layer_list[-1].get_output_size()

                self.add_layer(layer_list[-1])

        self.output_size = layer_list[-1].get_output_size()


    def run(self, X, dropout_keep_prob=1.):
        """
        Passes data through each layer in the layer list in the opposite order in which they were added.

        Input:

        X: (tensor) data to be passed through the processor. [batch, height, width, channels]
        """
        # Define while loop dependencies
        def run_conv_per_item(current_item, conv_output):
            conv_intermediate = tf.expand_dims(X[:, :, :, current_item], -1)
            for current_layer in self.layer_list[:self.num_siamese_layers]:
                if isinstance(current_layer, layer_instances.DropoutLayer):
                    conv_intermediate = current_layer.run(conv_intermediate, dropout_keep_prob=dropout_keep_prob)
                else:
                    conv_intermediate = current_layer.run(conv_intermediate)
            conv_output = conv_output.write(current_item, conv_intermediate)
            return conv_output

        # Run Simese
        current_item0 = 0
        conv_output0 = tf.TensorArray(tf.float32, size=self.num_items, dynamic_size=False, clear_after_read=False)
        # condition = lambda current_item, conv_output: current_item < self.num_items
        # body      = lambda current_item, conv_output: [current_item+1, run_conv_per_item(current_item, conv_output)]
        # fc_input = tf.while_loop(condition, body,
        #                                 loop_vars= [current_item0, conv_output0],
        #                                 parallel_iterations= self.num_items)
        # for iitem in range(self.num_items):
        #     if iitem == 0:
        #         fc_intermediate = fc_input[1].read(iitem)
        #     else:
        #         fc_intermediate = tf.concat([fc_intermediate, fc_input[1].read(iitem)], axis=3)

        if (self.organization == 'obj') | (self.organization == 'full'):
            repeat_for = self.num_items
        else:
            repeat_for = 1

        while current_item0 < repeat_for:
            conv_output0 = run_conv_per_item(current_item0, conv_output0)
            if current_item0 == 0:
                fc_intermediate = conv_output0.read(current_item0)
            else:
                fc_intermediate = tf.concat([fc_intermediate, conv_output0.read(current_item0)], axis=3)
            current_item0 += 1

        # SLICE
        # CONCAT IN BATCH DIM
        # SLICE AGAIN
        # Run FC
        for current_layer in self.layer_list[self.num_siamese_layers:]:
            if isinstance(current_layer, layer_instances.DropoutLayer):
                fc_intermediate = current_layer.run(fc_intermediate, dropout_keep_prob=dropout_keep_prob)
            else:
                fc_intermediate = current_layer.run(fc_intermediate)

        return fc_intermediate


class PSVRT_vgg19(processors.BaseFeedforwardProcessor):
    """
    A multilayer simese convolutional net. Input channels are processed separately using shared conv weights.
    """

    def initialize_vars(self):

        layer_list = []
        intermediate_output_size = self.get_output_size()

        # conv1_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv1_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=64, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv1_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv1_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=64, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool1
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool1', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv2_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv2_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=128, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv2_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv2_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=128, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool2
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool2', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv3_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=256, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv3_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=256, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv3_3
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_3', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=256, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv3_4
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_4', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=256, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool3
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool3', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv4_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv4_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv4_3
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_3', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv4_4
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_4', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool4
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool4', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv5_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv5_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv5_3
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_3', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv5_4
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_4', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool5
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool5', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # FC1
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC1', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=4096, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # Dropout1
        layer_list.append(layer_instances.DropoutLayer(name=self.name + '/Dropout1',
                                                       input_size=intermediate_output_size,
                                                       batch_size=self.batch_size))
        layer_list[-1].initialize_vars(dropout_multiplier=1.)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # FC2
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC2', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=4096, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # Dropout2
        layer_list.append(layer_instances.DropoutLayer(name=self.name + '/Dropout2',
                                                       input_size=intermediate_output_size,
                                                       batch_size=self.batch_size))
        layer_list[-1].initialize_vars(dropout_multiplier=1.)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # FC3
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC3', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=1000, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # Classification
        layer_list.append(layer_instances.FCLayer(name=self.name + '/classification', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=2, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        self.output_size = layer_list[-1].get_output_size()

    def run(self, X, dropout_keep_prob=1.):
        intermediate = X
        for current_layer in self.layer_list:
            if isinstance(current_layer, layer_instances.DropoutLayer):
                intermediate = current_layer.run(intermediate, dropout_keep_prob=dropout_keep_prob)
            else:
                intermediate = current_layer.run(intermediate)

        return intermediate


class PSVRT_vgg16(processors.BaseFeedforwardProcessor):
    """
    A multilayer simese convolutional net. Input channels are processed separately using shared conv weights.
    """

    def initialize_vars(self):
        use_bias_in_conv = False

        layer_list = []
        intermediate_output_size = self.get_output_size()

        # conv1_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv1_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=64, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv1_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv1_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=64, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # BN1
        layer_list.append(
            layer_instances.BatchNormLayer(name=self.name + '/bn1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars()
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool1
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool1', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv2_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv2_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=128, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv2_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv2_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=128, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # BN2
        layer_list.append(
            layer_instances.BatchNormLayer(name=self.name + '/bn2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars()
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool2
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool2', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv3_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=256, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv3_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=256, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv3_3
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_3', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=256, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # BN3
        layer_list.append(
            layer_instances.BatchNormLayer(name=self.name + '/bn3', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars()
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool3
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool3', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2, 2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv4_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv4_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv4_3
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_3', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # BN4
        layer_list.append(
            layer_instances.BatchNormLayer(name=self.name + '/bn4', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars()
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool4
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool4', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # conv5_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv5_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv5_3
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_3', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[3,3], output_channels=512, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # BN5
        layer_list.append(
            layer_instances.BatchNormLayer(name=self.name + '/bn5', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars()
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # pool5
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool5', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[2,2], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])

        # FC1
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC1', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=4096, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # Dropout1
        layer_list.append(layer_instances.DropoutLayer(name=self.name + '/Dropout1',
                                                       input_size=intermediate_output_size,
                                                       batch_size=self.batch_size))
        layer_list[-1].initialize_vars(dropout_multiplier=1.)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # FC2
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC2', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=4096, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # Dropout2
        layer_list.append(layer_instances.DropoutLayer(name=self.name + '/Dropout2',
                                                       input_size=intermediate_output_size,
                                                       batch_size=self.batch_size))
        layer_list[-1].initialize_vars(dropout_multiplier=1.)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # FC3
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC3', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=1000, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # Classification
        layer_list.append(layer_instances.FCLayer(name=self.name + '/classification', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=2, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        self.output_size = layer_list[-1].get_output_size()

    def run(self, X, dropout_keep_prob=1.):
        intermediate = X
        for current_layer in self.layer_list:
            if isinstance(current_layer, layer_instances.DropoutLayer):
                intermediate = current_layer.run(intermediate, dropout_keep_prob=dropout_keep_prob)
            else:
                intermediate = current_layer.run(intermediate)

        return intermediate


class PSVRT_deepcontrol(processors.BaseFeedforwardProcessor):
    """
    A multilayer simese convolutional net. Input channels are processed separately using shared conv weights.
    """

    def initialize_vars(self):
        use_bias_in_conv = False

        layer_list = []
        intermediate_output_size = self.get_output_size()

        # conv1_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv1_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[4,4], output_channels=8, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv1_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv1_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=8, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # # BN1
        # layer_list.append(
        #     layer_instances.BatchNormLayer(name=self.name + '/bn1', input_size=intermediate_output_size,
        #                                    batch_size=self.batch_size, trainable=True))
        # layer_list[-1].initialize_vars()
        # intermediate_output_size = layer_list[-1].get_output_size()
        # self.add_layer(layer_list[-1])
        # pool1
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool1', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[3,3], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])


        # conv2_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv2_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=16, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv2_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv2_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=16, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # # BN2
        # layer_list.append(
        #     layer_instances.BatchNormLayer(name=self.name + '/bn2', input_size=intermediate_output_size,
        #                                 batch_size=self.batch_size, trainable=True))
        # layer_list[-1].initialize_vars()
        # intermediate_output_size = layer_list[-1].get_output_size()
        # self.add_layer(layer_list[-1])
        # pool2
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool2', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[3,3], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])


        # conv3_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=32, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv3_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv3_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=32, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # # BN3
        # layer_list.append(
        #     layer_instances.BatchNormLayer(name=self.name + '/bn3', input_size=intermediate_output_size,
        #                                 batch_size=self.batch_size, trainable=True))
        # layer_list[-1].initialize_vars()
        # intermediate_output_size = layer_list[-1].get_output_size()
        # self.add_layer(layer_list[-1])
        # pool3
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool3', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[3,3], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])


        # conv4_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=64, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv4_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv4_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=64, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # # BN4
        # layer_list.append(
        #     layer_instances.BatchNormLayer(name=self.name + '/bn4', input_size=intermediate_output_size,
        #                                 batch_size=self.batch_size, trainable=True))
        # layer_list[-1].initialize_vars()
        # intermediate_output_size = layer_list[-1].get_output_size()
        # self.add_layer(layer_list[-1])
        # pool4
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool4', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[3,3], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])


        # con5_1
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_1', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=128, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # conv5_2
        layer_list.append(
            layer_instances.Conv2dLayer(name=self.name + '/conv5_2', input_size=intermediate_output_size,
                                        batch_size=self.batch_size, trainable=True))
        layer_list[-1].initialize_vars(rf_size=[2,2], output_channels=128, stride=[1, 1],
                                       activation_type='relu', keep_size=True, use_bias=use_bias_in_conv)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # # BN5
        # layer_list.append(
        #     layer_instances.BatchNormLayer(name=self.name + '/bn5', input_size=intermediate_output_size,
        #                                 batch_size=self.batch_size, trainable=True))
        # layer_list[-1].initialize_vars()
        # intermediate_output_size = layer_list[-1].get_output_size()
        # self.add_layer(layer_list[-1])
        # pool5
        layer_list.append(
            layer_instances.Maxpool2dLayer(name=self.name + '/pool5', input_size=intermediate_output_size,
                                  batch_size=self.batch_size))
        layer_list[-1].initialize_vars(rf_size=[3,3], stride=[2,2])
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])


        # FC1
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC1', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=256, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # Dropout1
        layer_list.append(layer_instances.DropoutLayer(name=self.name + '/Dropout1',
                                                       input_size=intermediate_output_size,
                                                       batch_size=self.batch_size))
        layer_list[-1].initialize_vars(dropout_multiplier=1.)
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # FC2
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC2', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=256, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # FC3
        layer_list.append(layer_instances.FCLayer(name=self.name + '/FC3', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=256, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        # Classification
        layer_list.append(layer_instances.FCLayer(name=self.name + '/classification', input_size=intermediate_output_size,
                                          batch_size=self.batch_size))
        layer_list[-1].initialize_vars(output_channels=2, activation_type='relu')
        intermediate_output_size = layer_list[-1].get_output_size()
        self.add_layer(layer_list[-1])
        self.output_size = layer_list[-1].get_output_size()

    def run(self, X, dropout_keep_prob=1.):
        intermediate = X
        for current_layer in self.layer_list:
            if isinstance(current_layer, layer_instances.DropoutLayer):
                intermediate = current_layer.run(intermediate, dropout_keep_prob=dropout_keep_prob)
            else:
                intermediate = current_layer.run(intermediate)

        return intermediate


class PSVRT_inception_v4(processors.BaseFeedforwardProcessor):
    """
    A multilayer simese convolutional net. Input channels are processed separately using shared conv weights.
    """

    def initialize_vars(self):
        print('inception_v4: initialize_vars nothing to do')
    #import inception_dependencies as sota

    def run(self, X, dropout_keep_prob=1.):
        #import ipdb
        #ipdb.set_trace()
        import inception_v4_dependencies as sota
        logits, end_points = \
            sota.inception_v4(X, num_classes=2, is_training=True,
                            dropout_keep_prob=dropout_keep_prob,
                            reuse=None,
                            scope='InceptionV4',
                            create_aux_logits=True)
        logits = tf.expand_dims(tf.expand_dims(logits, axis=1), axis=1)
        return logits

class PSVRT_inception_resnet_v2(processors.BaseFeedforwardProcessor):
    """
    A multilayer simese convolutional net. Input channels are processed separately using shared conv weights.
    """

    def initialize_vars(self):
        print('inception_resnet_v2: initialize_vars nothing to do')
        #import inception_dependencies as sota

    def run(self, X, dropout_keep_prob=1.):
        #import ipdb
        #ipdb.set_trace()
        import inception_v4_dependencies as sota
        logits, end_points = \
            sota.inception_v4(X, num_classes=2, is_training=True,
                              dropout_keep_prob=dropout_keep_prob,
                              reuse=None,
                              scope='Inception-resnet-V2',
                              create_aux_logits=True)
        logits = tf.expand_dims(tf.expand_dims(logits, axis=1), axis=1)
        return logits

class PSVRT_resnet50_v2(processors.BaseFeedforwardProcessor):
    """
    A multilayer simese convolutional net. Input channels are processed separately using shared conv weights.
    """

    def initialize_vars(self):
        print('inception_resnet_v2: initialize_vars nothing to do')
        # import inception_dependencies as sota

    def run(self, X, dropout_keep_prob=1.):
        # import ipdb
        # ipdb.set_trace()
        import resnet_v2_dependencies as sota
        logits, end_points = \
            sota.inception_v4(X, num_classes=2, is_training=True,
                              dropout_keep_prob=dropout_keep_prob,
                              reuse=None,
                              scope='Inception-resnet-V2',
                              create_aux_logits=True)
        logits = tf.expand_dims(tf.expand_dims(logits, axis=1), axis=1)
        return logits