import app.config

import sqlalchemy
from sqlalchemy import MetaData

instance = sqlalchemy.create_engine('postgresql:///tutorial.db')
instance.connect()

metadata = MetaData()
