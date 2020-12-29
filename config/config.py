import xml.dom.minidom as minidom

class config(object):
    def __init__(self):
        return

    @staticmethod
    def get_config(xml_path):
        dom_tree = minidom.parse(xml_path)
        collection = dom_tree.documentElement
        if collection.nodeName != 'config':
            raise RuntimeError('this is invalid nn config: the must has header "config"')

        config = dict()

        train = collection.getElementsByTagName('train')
        config['batch_size'] = int(train[0].getElementsByTagName('batch_size')[0].firstChild.data)
        config['eval_batch_size'] = int(train[0].getElementsByTagName('eval_batch_size')[0].firstChild.data)
        config['train_epoch'] = int(train[0].getElementsByTagName('train_epoch')[0].firstChild.data)
        config['device'] = str(train[0].getElementsByTagName('device')[0].firstChild.data)
        if config['device'] == 'None':
            config['device'] = None
        config['cpu_cores'] = str(train[0].getElementsByTagName('cpu_cores')[0].firstChild.data)
        if config['cpu_cores'] == 'None':
            config['cpu_cores'] = None
        else:
            config['cpu_cores'] = int(config['cpu_cores'])

        optimize = collection.getElementsByTagName('optimize')
        config['learning_rate'] = float(optimize[0].getElementsByTagName('learning_rate')[0].firstChild.data)
        config['end_learning_rate'] = float(optimize[0].getElementsByTagName('end_learning_rate')[0].firstChild.data)
        config['decay_rate'] = float(optimize[0].getElementsByTagName('decay_rate')[0].firstChild.data)
        config['epsilon'] = float(optimize[0].getElementsByTagName('epsilon')[0].firstChild.data)
        config['decay_steps'] = float(optimize[0].getElementsByTagName('decay_steps')[0].firstChild.data)
        config['update_mode_freq'] = int(optimize[0].getElementsByTagName('update_mode_freq')[0].firstChild.data)

        result = collection.getElementsByTagName('result')
        config['image_path'] = result[0].getElementsByTagName('image_path')[0].firstChild.data
        config['out_path'] = result[0].getElementsByTagName('out_path')[0].firstChild.data
        config['img_width'] = int(result[0].getElementsByTagName('img_width')[0].firstChild.data)
        config['img_height'] = int(result[0].getElementsByTagName('img_height')[0].firstChild.data)

        return config


