import xml.dom.minidom as minidom

class config(object):
    def __init__(self):
        return

    @staticmethod
    def get_value(handle, key):
        value = None
        tag_name = handle[0].getElementsByTagName(key)[0].firstChild
        if tag_name is not None:
            value = tag_name.data
            if value == 'None':
                value = None
        return value

    @staticmethod
    def get_config(xml_path):
        dom_tree = minidom.parse(xml_path)
        collection = dom_tree.documentElement
        if collection.nodeName != 'config':
            raise RuntimeError('this is invalid nn config: the must has header "config"')

        cf = dict()

        train = collection.getElementsByTagName('train')
        cf['batch_size'] = int(config.get_value(train, 'batch_size'))
        cf['eval_batch_size'] = int(config.get_value(train, 'eval_batch_size'))
        cf['train_epoch'] = int(config.get_value(train, 'train_epoch'))
        cf['device'] = config.get_value(train, 'device')
        cf['cpu_cores'] = config.get_value(train, 'cpu_cores')
        if cf['cpu_cores'] is not None:
            cf['cpu_cores'] = int(cf['cpu_cores'])

        optimize = collection.getElementsByTagName('optimize')
        cf['learning_rate'] = float(config.get_value(optimize, 'learning_rate'))
        cf['end_learning_rate'] = float(config.get_value(optimize, 'end_learning_rate'))
        cf['decay_rate'] = float(config.get_value(optimize, 'decay_rate'))
        cf['epsilon'] = float(config.get_value(optimize, 'epsilon'))
        cf['decay_steps'] = float(config.get_value(optimize, 'decay_steps'))
        cf['update_mode_freq'] = int(config.get_value(optimize, 'update_mode_freq'))
        cf['sim_loss_w'] = float(config.get_value(optimize, 'sim_loss_w'))
        cf['shp_loss_w'] = float(config.get_value(optimize, 'shp_loss_w'))
        cf['lanes'] = config.get_value(optimize, 'lanes')

        result = collection.getElementsByTagName('result')
        cf['image_path'] = config.get_value(result, 'image_path')
        cf['out_path'] = config.get_value(result, 'out_path')
        cf['img_width'] = int(config.get_value(result, 'img_width'))
        cf['img_height'] = int(config.get_value(result, 'img_height'))

        return cf


