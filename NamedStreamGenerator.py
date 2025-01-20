from strlearn.streams import StreamGenerator

class NamedStreamGenerator(StreamGenerator):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name