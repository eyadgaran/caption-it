'''
Module for database tables
'''

__author__ = 'Elisha Yadgaran'


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, func, BigInteger, MetaData
from sqlalchemy_mixins import AllFeaturesMixin


Base = declarative_base()


class BaseModel(Base, AllFeaturesMixin):
    __abstract__ = True
    metadata = MetaData()
    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), nullable=True, server_onupdate=func.now())
    id = Column(BigInteger, primary_key=True)


class Feedback(BaseModel):
    __tablename__ = 'feedback'

    feedback = Column(String(1200), nullable=False)


class ModelHistory(BaseModel):
    __tablename__ = 'model_history'

    filename = Column(String())
    caption = Column(String())
    user_rank = Column(String())
    user_caption = Column(String())
