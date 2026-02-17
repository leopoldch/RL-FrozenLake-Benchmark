from abc import ABC, abstractmethod

# import pour définir une classe abstraite

class Strategy(ABC):

    @abstractmethod
    def play(action):
        pass
