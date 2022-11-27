from .base_coupling_layer import BaseCouplingLayer


class AdditiveCouplingLayer(BaseCouplingLayer):
    def coupling_law(self, a, b):
        return a + b

    def anticoupling_law(self, a, b):
        return a - b
