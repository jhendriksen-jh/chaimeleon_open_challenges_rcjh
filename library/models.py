"""
Model class definitions
"""


class ChaimeleonChallengeModel:
    def __init__(self):
        pass

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
