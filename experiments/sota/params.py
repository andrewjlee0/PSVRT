import os, sys
sys.path.append(os.path.abspath(os.path.join('..','..')))

def get_params():
    params = {}
    params['raw_input_size'] = [180, 180, 1]
    params['batch_size'] = 50 #6
    params['num_categories'] = 2
    params['num_min_train_imgs'] = 5000000 # 5M shorter run
    params['num_max_train_imgs'] = 5000000
    params['num_val_period_imgs'] = params['num_max_train_imgs']/400
    params['num_val_imgs'] = 500
    params['threshold_loss'] = 1.1

    params['learning_rate'] = 1e-4
    params['clip_gradient'] = True
    params['dropout_keep_prob'] = 0.8

    from instances import processor_instances

    params['model_obj'] = processor_instances.PSVRT_inception_resnet_v2
    params['model_name'] = 'model'
    params['model_init_args'] = {}
    from instances import psvrt
    params['train_data_obj'] = psvrt.psvrt
    params['train_data_init_args'] = {'problem_type':'SD',
                                      'item_size': [4,4],
                                      'box_extent': [60,60],
                                      'num_items': 2,
                                      'num_item_pixel_values': 1,
                                      'display': False}

    params['val_data_obj'] = psvrt.psvrt
    params['val_data_init_args'] = params['train_data_init_args'].copy()

    params['learningcurve_type'] = 'array'

    return params
