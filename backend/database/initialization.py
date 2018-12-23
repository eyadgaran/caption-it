'''
Module for database initialization
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils.initialization import Database
from backend.database.models import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import os
from os.path import join, dirname, abspath
import ConfigParser


def parse_cnf(cnf_section):
    '''
    Assumes there is a cnf file located at ~/.my.cnf with sections and parameters
    :param cnf_section: cnf section title for param group
    :return: dictionary of parameters (None defaults to empty string)
    '''
    config = ConfigParser.SafeConfigParser(allow_no_value=True)
    config.read(os.getenv('CNF_FILE',
                          join(dirname(abspath(__file__)), 'secret.cnf')))
    parameter_dict = dict(config.items(cnf_section))

    return parameter_dict


class SimpleMLDatabase(object):
    def __init__(self):
        self.database_params = parse_cnf('captioner-simpleml')

    def initialize(self, **kwargs):
        db = Database(**self.database_params)
        db.initialize(**kwargs)


class AppDatabase(object):
    def __init__(self):
        self.database_params = parse_cnf('captioner-app')

    def initialize(self):
        url = '{jdbc}://{user}:{password}@{host}:{port}/{database}'.format(
            **self.database_params)
        engine = create_engine(url)
        db_session = scoped_session(sessionmaker(autocommit=True,
                                                 autoflush=False,
                                                 bind=engine))
        # Create tables.
        BaseModel.metadata.create_all(bind=engine)
        BaseModel.set_session(db_session)
