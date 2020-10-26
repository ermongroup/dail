import tensorflow as tf


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

def get_initializer(name_list):
    '''
    Converts list of initializer names into tf initialization function
    Args:
        name: list of string format names for the initializer
    Returns:
        inits: list of tf initialization functions
    '''
    inits = []
    for n_ in name_list:
        if n_ is None:
            inits.append(None)
        elif n_ == 'he':
            inits.append(tf.contrib.layers.variance_scaling_initializer())
        elif n_ == 'xavier':
            inits.append(tf.contrib.layers.xavier_initializer())
        elif n_ == 'normal':
            inits.append(tf.initializers.truncated_normal(mean=0., stddev=0.1))
        else:
            print("Unrecognized initializer class: {}".format(n_))
            exit(1)

    return inits


def get_regularizer(name_list, scale=[0.001]):
    '''
    Converts list of regularizer names into tf regularization function
    Args:
        name_list: list of string format names for the regularizer
    Returns:
        regularizers: list of tf regularization functions
    '''
    regularizers = []
    for i, n_ in enumerate(name_list):
        if n_ is None:
            regularizers.append(None)
        elif n_ == 'l2':
            regularizers.append(tf.contrib.layers.l2_regularizer(scale=scale[i]))
        elif n_ == 'l1':
            regularizers.append(tf.contrib.layers.l1_regularizer(scale=scale[i]))
        elif n_ == 'l1_l2':
            ### TODO: handle different scales for l1, l2 regularizers
            regularizers.append(tf.contrib.layers.l1_l2_regularizer(scale_l1=scale[i], scale_l2=scale[i]))
        else:
            print("Unrecognized regularizer class: {}".format(n_))
            exit(1)

    return regularizers


def get_activation(name_list):
    '''
    Converts list of activation names into tf activation function
    Args:
        name_list: list of string format names for the activation function
    Returns:
        activation: list of tf activation functions
    '''
    activation = []

    for n_ in name_list:
        if n_ is None:
            activation.append(None)
        elif n_ == 'relu':
            activation.append(tf.nn.relu)
        elif n_ == 'leaky_relu':
            activation.append(tf.nn.leaky_relu)
        elif n_ == 'sigmoid':
            activation.append(tf.sigmoid)
        else:
            print("Unrecognized activation class: {}".format(n_))
            exit(1)

    return activation


def scale_action(action, env):
    '''
    Scale actions to valid range
    Args:
        action: unscaled action tensor [batch, action_dim]
        env: OpenAI gym environment

    Returns:
        scaled_action: scaled action to the valid range of the environment
    '''
    scaled_action = env.action_space.low + tf.nn.sigmoid(action)*(env.action_space.high - env.action_space.low)
    return scaled_action


def scale_state(state, env):
    '''
    Scale states to valid range
    Args:
        state: unscaled state tensor [batch, state_dim]
        env: OpenAI gym environment

    Returns:
        scaled_state: scaled action to the valid range of the environment
    '''
    scaled_state = env.observation_space.low + tf.nn.sigmoid(state)*(env.observation_space.high - env.observation_space.low)
    return scaled_state

def feedforward(in_node, is_training, params, scope, scale=False, scale_fn=None, scale_params=None):
    '''
    Generic constructor for feedforward networks
    Args:
        in_node: input node (e.g. placeholders, previous hidden state)
        is_training: training flag for drop out
        params: network parameters
        scope: tf variable scope
    Returns:
        out_node: output node
    '''

    num_hidden = params['num_hidden']
    depth = len(num_hidden)

    # Check whether params is formatted properly or not
    assert depth == len(params['activation']) == len(params['init']) == len(params['regularizer'])
    assert depth == len(params['reg_scale'])

    activation = get_activation(name_list=params['activation'])
    init = get_initializer(name_list=params['init'])
    reg = get_regularizer(name_list=params['regularizer'], scale=params['reg_scale'])

    # Construct the feedforward network
    layers = [in_node]
    with tf.variable_scope(scope):
        for i in range(depth):
            cur_layer = tf.layers.dense(layers[-1], num_hidden[i],
                                        name='dense_'+str(i), activation=activation[i],
                                        kernel_initializer=init[i], kernel_regularizer=reg[i])
            layers.append(cur_layer)

    # Output node is the last layer of the network
    out_node = layers[-1]

    # Apply scaling
    if scale:
        out_node = scale_fn(out_node, scale_params)

    return out_node


def convnet(in_node, is_training, params, scope):
    '''
    Generic constructor for convolutional networks
    Args:
        in_node: input node (e.g. placeholders, previous hidden state)
        is_training: training flag for drop out
        params: network parameters
        scope: tf variable scope
    Returns:
        out_node: output node
    '''
    raise NotImplementedError
