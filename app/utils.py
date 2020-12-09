from sqlalchemy.ext.declarative import DeclarativeMeta
import json


def get_film_id(id):
    """Pad film ID to be 10 characters in length"""
    return "{:<10}".format(id)


def get_person_id(id):
    """Pad person ID to be 10 characters in length"""
    return "{:<14}".format(id)


class AlchemyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj.__class__, DeclarativeMeta):
            # an SQLAlchemy class
            fields = {}
            for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                data = obj.__getattribute__(field)
                try:
                    # this will fail on non-encodable values, like other classes
                    json.dumps(data)
                    fields[field] = data
                except TypeError:
                    fields[field] = None
            # a json-encodable dict
            return fields

        return json.JSONEncoder.default(self, obj)
