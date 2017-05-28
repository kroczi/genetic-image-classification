#! /usr/bin/python3
try:
    import configparser
except ImportError:
    import ConfigParser as configparser
import logging

logger = logging.getLogger('ic')


class ConfigSection:
    def __init__(self, config, section):
        self.config = config
        self.section = section

    def get(self, key):
        return self.config.get(self.section, key)

    def getint(self, key):
        return self.config.getint(self.section, key)

    def getfloat(self, key):
        return self.config.getfloat(self.section, key)

    def getboolean(self, key):
        return self.config.getboolean(self.section, key)


def acquire_configuration(dataset_config_file, parameters_config_file, dataset_profile, parameters_profile=None):
    dataset_configuration = configparser.ConfigParser()
    dataset_configuration.read(dataset_config_file)
    if dataset_configuration.has_section(dataset_profile):
        logger.debug("Loading dataset configuration for profile: " + dataset_profile)
        dataset_config = ConfigSection(dataset_configuration, dataset_profile)
    else:
        logger.error("!!!Dataset configuration not found!!!")
        raise KeyError

    parameters_configuration = configparser.ConfigParser()
    parameters_configuration.read(parameters_config_file)
    if parameters_configuration.has_section(parameters_profile):
        logger.debug("Loading parameters configuration for profile: " + parameters_profile)
        parameters_config = ConfigSection(parameters_configuration, parameters_profile)
    else:
        logger.debug("Loading parameters configuration from defaults.")
        parameters_config = ConfigSection(parameters_configuration, 'DEFAULT')

    return dataset_config, parameters_config
