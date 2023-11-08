class Tensor:
    def __init__(self, index: str = None, mat: list = None):
        self.matrix = mat
        self.index = index

    # def get(self, index: str):
    #     self.index = index
    #     if any(character.isalpha() for character in self.index):
    #         return self.copy()
    #     else:
    #         res = self.matrix
    #         for i in index:
    #             res = res[int(i)]
    #         return res

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
        mat = self.matrix
        while (condition):
            res.append(len(mat))
            if isinstance(mat[0], list):
                mat = mat[0]
            else:
                break
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
                # print(num_index)
                index_individual += [out.index[j:j + num_index]]
                j += num_index
                # print(index_individual)
            i = 0
            res = 1
            # print(index_individual)
            for tensor in out.tensors:
                temp_tensor = tensor.copy()
                # print(index_individual[i])
                res *= temp_tensor.get(index_individual[i])
                i += 1
            return res

    def contraction(self, contra_ind):
        res = [self]
        temp = []
        for ind in contra_ind:
            dim = self.shape()[self.index.find(ind)]
            for term in res:
                for i in range(dim):
                    t_index = term.index
                    temp += [term.element_product(t_index.replace(ind, str(i)))]
            res, temp = temp, []
        # print(res)
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


# def generating_matrix(shape: list, *position):
#     if not shape:
#         return str(position)
#     return [generating_matrix(shape[1:], *position, [i]) for i in range(shape[0])]


def generating_matrix(t: multiTensor, out_shape: list, target: str, contra_letters: str, *position):
    if not out_shape:
        out = t.copy()
        for letter, location in zip(target, position):
            out.index = out.index.replace(letter, str(location[0]))
            # print(location)
        # print(out.index)
        return out.contraction(contra_letters)
    return [generating_matrix(t, out_shape[1:], target, contra_letters, *position, [i]) for i in range(out_shape[0])]


def check(t: multiTensor):
    condition = True
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
    if len(t.index)<len(shape):
        raise KeyError('Missing index')
    elif len(t.index)>len(shape):
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
    for dim, letter in zip(raw_shape, raw_product.index):
        if letter in target:
            shape += [dim]
        # print(shape,'shape')
    # print(operation,shape)
    return generating_matrix(raw_product, shape, target, summation)

# def contraction(t: Tensor, contra_ind): # good
#     res = [t]
#     temp = []
#     for ind in contra_ind:
#         dim = t.shape()[t.index.find(ind)]
#         for term in res:
#             for i in range(dim):
#                 t_index = term.index
#                 temp += [term.get(t_index.replace(ind, str(i)))]
#         res, temp = temp, []
#     # print(res)
#     return sum(res)


# def multi_mat(index: str, *tensors: Tensor):
#     copy = []
#     j = 0
#     index_individual = []
#     for tensor in tensors:
#         num_index = len(tensor.shape())
#         # print(num_index)
#         index_individual += [index[j:j + num_index]]
#         j += num_index
#         # print(index_individual)
#     i = 0
#     res = 1
#     # print(index_individual)
#     for tensor in tensors:
#         temp_tensor = tensor.copy()
#         res *= temp_tensor.get(index_individual[i])
#         i += 1
#     return res

# d = Tensor('ab', [1, 2])
# a = [[1, 2], [3, 4]]
# b = Tensor('ab', a)
# print(b.get('01'))
# print(b.get('i0').index)
# print(b.shape())
# print(contraction(b, 'i'))
# print(multi_mat('0011', b, b))
# c = multiTensor('ijkj', b, b)
# d=multiTensor('aiiiiiia',d,*c.tensors,b,d)
# print(multiTensor('a1b', b, d).contraction('ab'))
# print(classify('ij,jk,kl->il'))
# print(einsum('iiii->',a,a))
# a = np.arange(25).reshape(5,5)
# b = np.arange(5)
# c = np.arange(6).reshape(2,3)
#
# # Test Case 1: Trace of a matrix
# result = einsum('ii->', a.tolist())
# expected = np.einsum('ii->', a)
# print(result,expected)
