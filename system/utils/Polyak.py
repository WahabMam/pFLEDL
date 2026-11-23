import copy

class PolyakBase:
    def __init__(self, mix, model):
        self.num_weights = 0
        self.mix = mix
        self.weight_sum = self.zero_parameters(model)

    def add_weight(self, model):
        self.num_weights += 1
        self.weight_sum = self.add_parameters(
            self.weight_sum, 
            model
        )
    
    def polyak_ruppert_averaging(self):
        return self.divide_constant(self.weight_sum, self.num_weights)

    def models_mixture(self, old, new):
        for new_param, old_param, avg in zip(new.parameters(), old.parameters(), self.polyak_ruppert_averaging().parameters()):
            old_param.data = new_param.data.clone() - self.mix * (new_param.data.clone() - avg.data.clone())  
        return old
    
    def multiply_constant(self, model, factor):
        result = copy.deepcopy(model)
        for r, param in zip(result.parameters(), model.parameters()):
            r.data = param.data.clone() * factor
        return result

    def divide_constant(self, model, divisor):
        result = copy.deepcopy(model)
        for r, param in zip(result.parameters(), model.parameters()):
            r.data = param.data.clone() / divisor
        return result
    
    def add_parameters(self, model1, model2):
        result = copy.deepcopy(model1)
        for r, param1, param2 in zip(result.parameters(), model1.parameters(), model2.parameters()):
            r.data = param1.data.clone() + param2.data.clone()
        return result
    
    def subtract_parameters(self, model1, model2):
        result = copy.deepcopy(model1)
        for r, param1, param2 in zip(result.parameters(), model1.parameters(), model2.parameters()):
            r.data = param1.data.clone() - param2.data.clone()
        return result

    def zero_parameters(self, model):
        zero = copy.deepcopy(model)
        for param in zero.parameters():
            param.data.zero_()
        return zero
    
class Polyak(PolyakBase):
    def __init__(self, mix, model, freq):
        super().__init__(mix, model)
        self.freq = freq
        self.times = 0
        self.iter_choosen = []
    
    def add_weight(self, model):
        self.times += 1
        if self.times % self.freq == 0: 
            super().add_weight(model)
            self.iter_choosen.append(self.times)
