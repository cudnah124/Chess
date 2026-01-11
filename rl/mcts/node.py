import numpy as np


class MCTSNode:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
    
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
