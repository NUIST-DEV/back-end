from abc import abstractmethod

class IModel:
    '''process will be called by the controller'''
    @abstractmethod
    def process(self, data):
        pass

    def set_device(self, device):
        self.device = device
        # if self has a model, set the device for the model
        if hasattr(self, 'model'):
            self.model.to(self.device)
