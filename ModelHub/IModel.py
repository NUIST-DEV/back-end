from abc import abstractmethod

class IModel:
    '''process will be called by the controller'''
    @abstractmethod
    def process(self, data):
        pass

    @abstractmethod
    def predict(self, tensor):
        pass
