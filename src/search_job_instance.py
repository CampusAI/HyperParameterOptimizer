from abc import ABC, abstractmethod

class SearchJobInstance(ABC):
    """Abstract class linked to a single job.
       It is used to manage 
    """
    def __init__(self, id):
        self.id = id

    @abstractmethod
    def launch(self, **kwargs) -> int:
        """Execute command given the objective arguments
           IMPORTANT: Must be non-blocking!

           Returns:
            status(Int): 0 everything is ok, 1 there was some error
        """
        self.passed_args = kwargs
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> float:
        """Return final result of the optimization
        """
        raise NotImplementedError

    @abstractmethod
    def done(self) -> bool:
        """True if job has finished, false otherwise
           IMPORTANT: Must be non-blocking!
        """
        raise NotImplementedError

    @abstractmethod
    def kill(self) -> None:
        """Finish job
        """
        raise NotImplementedError

    @abstractmethod
    def end(self) -> None:
        """Run any task necessary when done
        """
        raise NotImplementedError