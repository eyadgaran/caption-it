'''
Module for database initialization
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils import Database, BaseDatabase
from .models import BaseModel


class SimpleMLDatabase(Database):
    def __init__(self):
        super(SimpleMLDatabase, self).__init__(configuration_section='simpleml-captioner')


class AppDatabase(BaseDatabase):
    def __init__(self):
        super(AppDatabase, self).__init__(configuration_section='app-captioner')

    def initialize(self, **kwargs):
        super(AppDatabase, self).initialize([BaseModel], create_tables=True, **kwargs)
