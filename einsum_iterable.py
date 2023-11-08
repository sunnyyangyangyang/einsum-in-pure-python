import typing


class Tensor:
    def __init__(self, index: str = None, mat: typing.Iterable = None):
        self.matrix = mat
        self.index = index

    def get(self, index: str):
        out = self.copy()
        out.index = index
        if any(character.isalpha() for character in out.index):
            return out
        else:
            res = out.matrix
            for i in index:
                res = res[int(i)]
            return res

    def shape(self) -> list:
        condition = True
        res = []
        mat = self.copy().matrix

        while condition:
            res.append(len(mat))
            if isinstance(mat[0],typing.Iterable):
                mat = mat[0]
            else:
                condition = False
        return res

    def copy(self):
        return Tensor(self.index, self.matrix)


class multiTensor():
    def __init__(self, index: str, *tensors: Tensor):
        self.tensors = tensors
        self.index = index

    def shape(self) -> list:
        res = []
        for tensor in self.tensors:
            res += tensor.shape()
        return res

    def element_product(self, index: str):
        out = self.copy()
        out.index = index
        if any(character.isalpha() for character in out.index):
            return out
        else:
            j = 0
            index_individual = []
            for tensor in out.tensors:
                num_index = len(tensor.shape())
                index_individual += [out.index[j:j + num_index]]
                j += num_index
            i = 0
            res = 1
            for tensor in out.tensors:
                temp_tensor = tensor.copy()
                res *= temp_tensor.get(index_individual[i])
                i += 1
            return res

    def contraction(self, contra_ind):
        # print(self.tensors,contra_ind)
        if not contra_ind:
            return self.element_product(self.index)
        res = [self]
        temp = []
        for ind in contra_ind:
            dim = self.shape()[self.index.find(ind)]
            for term in res:
                for i in range(dim):
                    t_index = term.index
                    temp += [term.element_product(t_index.replace(ind, str(i)))]
            res, temp = temp, []
        return sum(res)

    def copy(self):
        return multiTensor(self.index, *self.tensors)


def classify(input: str) -> tuple:
    divide = input.find('->')
    target = input[divide + 2:]
    temp = input[:divide]
    temp = temp.split(',')
    operate = ''.join(temp)
    summation = ''.join(list(set(operate) - set(target)))
    return operate, target, summation


def generating_matrix(t: multiTensor, out_shape: list, target: str, contra_letters: str, *position):
    if not out_shape:
        out = t.copy()
        for letter, location in zip(target, position):
            out.index = out.index.replace(letter, str(location[0]))
        return out.contraction(contra_letters)
    return [generating_matrix(t, out_shape[1:], target, contra_letters, *position, [i]) for i in range(out_shape[0])]


def check(t: multiTensor):
    shape = t.shape()
    wrong_keys = []
    for letter in set(t.index):
        specific_shape = []
        for i, s in zip(t.index, shape):
            if i == letter:
                specific_shape += [s]
        if not max(specific_shape) == min(specific_shape):
            wrong_keys += [letter]
    if wrong_keys:
        raise KeyError(f'Index not match the shape:{wrong_keys}')
    if len(t.index) < len(shape):
        raise KeyError('Missing index')
    elif len(t.index) > len(shape):
        raise KeyError('Extra index found')


def einsum(operation: str, *matrices) -> list:
    operate, target, summation = classify(operation)
    raw_tensors = []
    for matrix in matrices:
        raw_tensors += [Tensor(mat=matrix)]
    raw_product = multiTensor(operate, *raw_tensors)
    check(raw_product)
    raw_shape = raw_product.shape()
    shape = []
    for letter in target:
        for dim, l in zip(raw_shape, raw_product.index):
            if l == letter:
                shape += [dim]
    return generating_matrix(raw_product, shape, target, summation)
