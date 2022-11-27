"""
Defines an abstract class for pose estimators
"""
import abc

from pose_estimation.human_pose import HumanPose

class BaseEstimator(metaclass=abc.ABCMeta):
    """
    Abstract class for any shell communication (can be local or through SSH)
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        """
        
        """
        methods = [
            'find_pose'
        ]
        for i in methods:
            if not hasattr(subclass, i) or not callable(subclass.__dict__[i]):
                return NotImplemented
        return True

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def find_pose(self, image) -> HumanPose:
        raise NotImplementedError
