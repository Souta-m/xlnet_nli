#!/usr/bin/env python3

import logging


def get_logger(identifier, base_path=''):
    console_log = logging.StreamHandler()
    logging.getLogger('pytorch_transformers.modeling_utils').addHandler(console_log)
    logging.getLogger('pytorch_transformers.modeling_utils').setLevel(logging.INFO)

    logger = logging.getLogger(identifier)
    hdlr = logging.FileHandler(base_path + 'logs/{}.log'.format(identifier))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    hdlr.setFormatter(formatter)
    console_log.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(console_log)
    logger.setLevel(logging.INFO)
    return logger
