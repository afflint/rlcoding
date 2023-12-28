from abc import abstractmethod


class Policy:
    def __init__(self, mdp) -> None:
        self.mdp = mdp
    
    @abstractmethod
    def __getitem__(self, state: int) -> int:
        """
        Get action

        Args:
            state (int): State

        Returns:
            int: The action chosen
        """
        pass

class StationaryPolicy(Policy):
    def __init__(self, mdp, state_action: dict) -> None:
        super().__init__(mdp)
        self.state_action = state_action
    
    def __getitem__(self, state: int) -> int:
        return self.state_action[state]