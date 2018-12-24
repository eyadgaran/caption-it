from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, DateTime, func, BigInteger
from sqlalchemy_mixins import AllFeaturesMixin


Base = declarative_base()


class BaseModel(Base, AllFeaturesMixin):
    __abstract__ = True
    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), nullable=True, server_onupdate=func.now())
    id = Column(BigInteger, primary_key=True)
