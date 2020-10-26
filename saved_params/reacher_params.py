def generate_params(env):
        '''
        Generates experiment parameters
        Args:
                env : environments for all domains : dict
        Returns:
                params : parameters : dict
        '''

        train_params = {'use_inclusive_graph': True,
                                        'gamma': 0.99,
                                        'num_episodes': 1000000,
                                        'max_steps_ep': 10000,
                                        'tau': 1e-2,
                                        'train_every': 1,
                                        'initial_noise_scale': 0.2, # TODO: Change this back to 1. (4 for mountaincar expert training)
                                        'noise_decay': 0.99,
                                        'exploration_mu': 0.0,
                                        'exploration_theta': 0.15,
                                        'exploration_sigma': 0.2,
                                        'memtype': 'vanilla',
                                        'memsize': int(1e5), #1e5
                                        'expert_memsize': int(1e5),
                                        'learner_memsize': int(1e4),
                                        'batchsize': 256, #256
                                        'scale_state': False,
                                        'scale_action': False,
                                        'tloss_horizon': 1,
                                        'use_wgan': False,
                                        'use_grad_wgan': False,
                                        'max_set_size': 1,
                                        'eps_decay_rate': 5e-5,
                                        'min_eps': 0.1}

        '''
        bc_params = {'num_epochs': 10000,
                                 'batches_per_epoch': 1000,
                                 'batchsize': 64,
                                 'lr': 1e-4}
        '''
        bc_params = {'num_epochs': 100,
                     'batches_per_epoch': 10000,
                     'batchsize': 256,
                     'lr': 1e-3}

#        reg_scale = 1e-7
        reg_scale = 0.
        #l1_reg_scale = 1e-6
        l1_reg_scale = 1e-8
        init = 'he'
        h = 64
        lr_decay = 1. ## TODO: CHANGE THIS BACK TO 1.
        act = 'leaky_relu' # leaky_relu

        # snake3 => snake 3: for expert: 1e-3, noise: 1. for learner: 5e-5, noise: 1., action_loss + 0.01*gen_loss
        # snake3(l) => snake 4(e): for expert: 1e-3, noise: 0.5**(key). for learner: 5e-5, noise: 1., action_loss + 0.01*gen_loss
        # snake4(l) => snake 3(e): for expert: 1e-3, noise: 1., for learner: 2e-5, noise: 1., action_loss + 0.01*gen_loss
        # snake3(l) => snake 5(e): for expert: 1e-3, noise: 0.5**(key). for learner: 2e-5, noise: 1., action_loss + 0.01*gen_loss
        # snake3(l) => snake 7(e):


        ######## FIGURING OUT HOW TO TRAIN CONSTRAINED POLICY ROBUSTLY FIRST => gen_loss set to 0 ####

        actlr_expert = 1e-4 ## 5e-5 good for snake3 learner,
        actlr_learner = 1e-5
        criticlr = 1e-3

        modellr = 1e-3
        disclr = 1e-3
        setlr = 1e-3

        autolr = 1e-4
        statelr = 1e-3
        actionlr = 1e-3

        goal_dim = 2

        expert_params = {'actor': {'lr': actlr_expert,
                                   'lr_decay': lr_decay,
                                   'num_hidden': [300, 200] + [env['expert']['action_dim']],
                                   'activation': [act]*2 + [None],
                                   'init': [init]*3,
                                   'regularizer': [None]*3,
                                   'reg_scale': [reg_scale]*3},

                         'critic': {'lr': criticlr,
                                    'lr_decay': lr_decay,
                                    'num_hidden': [400, 300] + [1],
                                    'activation': [act]*2 + [None],
                                    'init': [init]*3,
                                    'regularizer': [None]*3,
                                    'reg_scale': [reg_scale]*3},

                         'statemap': {'lr': statelr,
                                      'lr_decay': 1.,
                                      'num_hidden': [h]*2 + [env['learner']['state_dim']],
                                      'activation': [act]*2 + [None],
                                      'init': [init]*3,
                                      'regularizer': [None]*3,
                                      'reg_scale': [reg_scale]*3},

                         'actionmap': {'lr': actionlr,
                                       'lr_decay': 1.,
                                       'num_hidden': [h]*2 + [env['expert']['action_dim']],
                                       'activation': [act]*2 + [None],
                                       'init': [init]*3,
                                       'regularizer': [None]*3,
                                       'reg_scale': [reg_scale]*3},

                         'setsizemap': {'lr': setlr,
                                        'lr_decay': 1.,
                                        'num_hidden': [h]*2 + [train_params['max_set_size']],
                                        'activation': [act]*2 + [None],
                                        'init': [init]*3,
                                        'regularizer': [None]*3,
                                        'reg_scale': [l1_reg_scale]*3},

                         'model': {'lr': modellr,
                                   'lr_decay': lr_decay,
                                   'num_hidden': [h]*3 + [env['expert']['state_dim']-goal_dim],
                                   'activation': [act]*3 + [None],
                                   'init': [init]*4,
                                   'regularizer': [None]*4,
                                   'reg_scale': [reg_scale]*4},

                         'discriminator': {'lr': disclr,
                                           'lr_decay': lr_decay,
                                           'num_hidden': [h]*2 + [1],
                                           'activation': [act]*2 + [None],
                                           'init': [init]*3,
                                           'regularizer': [None]*3,
                                           'reg_scale': [reg_scale]*3},

                         'auto': {'lr': autolr,
                                  'lr_decay': 1.}
                        }

        learner_params = {'use_bc': True,
                          'actor': {'lr': actlr_learner,
                                    'lr_decay': lr_decay,
                                    'num_hidden': [300, 200] + [env['learner']['action_dim']],
                                    'activation': [act]*2 + [None],
                                    'init': [init]*3,
                                    'regularizer': [None]*3,
                                    'reg_scale': [reg_scale]*3},

                          'critic': {'lr': criticlr,
                                     'lr_decay': lr_decay,
                                     'num_hidden': [400, 300] + [1],
                                     'activation': [act]*2 + [None],
                                     'init': [init]*3,
                                     'regularizer': ['l2']*3,
                                     'reg_scale': [reg_scale]*3},

                          'statemap': {'lr': statelr,
                                       'lr_decay': 1.,
                                       'num_hidden': [200]*2 + [env['expert']['state_dim']-goal_dim],
                                       'activation': [act]*2 + [None],
                                       'init': [init]*3,
                                       'regularizer': ['l2']*3,
                                       'reg_scale': [reg_scale]*3},

                          'invmap': {'lr': autolr,
                                     'lr_decay': 1.,
                                     'num_hidden': [200]*2 + [env['learner']['state_dim']-goal_dim],
                                     'activation': [act]*2 + [None],
                                     'init': [init]*3,
                                     'regularizer': ['l2']*3,
                                     'reg_scale': [reg_scale]*3},

                          'actionmap': {'lr': actionlr,
                                        'lr_decay': 1.,
                                        'num_hidden': [200]*2 + [env['learner']['action_dim']],
                                        'activation': [act]*2 + [None],
                                        'init': [init]*3,
                                        'regularizer': ['l2']*3,
                                        'reg_scale': [reg_scale]*3},

                          'setsizemap': {'lr': setlr,
                                         'lr_decay': 1.,
                                         'num_hidden': [h]*2 + [train_params['max_set_size']],
                                         'activation': [act]*2 + [None],
                                         'init': [init]*3,
                                         'regularizer': [None]*3,
                                         'reg_scale': [l1_reg_scale]*3},

                          'model': {'lr': modellr,
                                    'lr_decay': lr_decay,
                                    'num_hidden': [64]*3 + [env['learner']['state_dim']-goal_dim],
                                    'activation': [act]*3 + [None],
                                    'init': [init]*4,
                                    'regularizer': [None]*4,
                                    'reg_scale': [reg_scale]*4},

                          'discriminator': {'lr': disclr,
                                            'lr_decay': lr_decay,
                                            'num_hidden': [64]*2 + [1],
                                            'activation': [act]*2 + [None],
                                            'init': [init]*3,
                                            'regularizer': [None]*3,
                                            'reg_scale': [reg_scale]*3},

                          'auto': {'lr': autolr,
                                   'lr_decay': 1.}
                        }


        params = {'train': train_params,
                  'expert': expert_params,
                  'learner': learner_params,
                  'bc': bc_params}

        return params
