from logging import warn

class Defaults(object):
    encoding = 'utf-8'
    token_regex = r'(\S+)'
    random_seed = 0xC001533D
    
    def __getattr__(self, name):
        warn('missing default for {}'.format(name))
        setattr(self, name, None)
        return None

defaults = Defaults()
