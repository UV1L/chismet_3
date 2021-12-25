import math
import numpy


class Method:

    def __init__(self):
        self.A, self.b = Utils.get_args()
        self.length = len(self.A)


class Utils:

    @staticmethod
    def get_args():
        A = [[-0.10, 1.00, -0.10],
                         [1.49, -0.20, 0.10],
                         [-0.30, 0.20, -0.50]]

        b = [1.63, 1.39, -1.40]

        return A, b


class Gauss1(Method):

    # Метод Гаусса
    def execute(self):

        self.first_iter()
        self.second_iter()

        return self.b

    def first_iter(self):
        for i in range(self.length):

            a = self.A[i][i]
            for j in range(i, self.length):
                self.A[i][j] = self.A[i][j] / a

            self.b[i] = self.b[i] / a
            for j in range(i + 1, self.length):

                c = -self.A[j][i]
                for k in range(i, self.length):
                    self.A[j][k] += c * self.A[i][k]
                self.b[j] += c * self.b[i]

    def second_iter(self):
        for i in range(self.length - 1, -1, -1):

            for k in range(i - 1, -1, -1):
                c = -self.A[k][i]
                self.A[k][i] += c * self.A[i][i]
                self.b[k] += c * self.b[i]


class Gauss2(Method):

    def __init__(self):
        super().__init__()
        self.A = numpy.array(self.A)
        self.b = numpy.array(self.b)

    # Метод Гаусса с выбором главного элемента
    def execute(self):
        for k in range(self.length - 1):

            max_elem = 0
            str = 0
            for i in range(k, self.length):
                if abs(self.A[i, k]) > abs(max_elem):
                    max_elem = self.A[i, k]
                    str = i

            change = numpy.repeat(self.A[k], 1)
            self.A[k], self.A[str] = self.A[str], change

            change = numpy.repeat(self.b[k], 1)
            self.b[k], self.b[str] = self.b[str], change

            self.A[k] = self.A[k] / max_elem
            self.b[k] = self.b[k] / max_elem

            for i in range(k + 1, self.length):
                factor = self.A[i, k]
                self.A[i] = self.A[i] - self.A[k] * factor
                self.b[i] = self.b[i] - self.b[k] * factor

        arg = [self.b[self.b.shape[0] - 1] / (self.A[self.length - 1, self.length - 1])]
        for i in range(self.length - 2, -1, -1):
            n = self.b[i]
            for j in range(len(arg)):
                n = n - arg[j] * self.A[i, self.length - 1 - j]
            arg.append(n)

        result = []
        for i in reversed(arg):
            result.append(i)

        return result


def error(x, y):
    v1 = (x[0] - y[0]) ** 2
    v2 = (x[1] - y[1]) ** 2
    v3 = (x[2] - y[2]) ** 2

    result = math.sqrt(v1 + v2 + v3)
    return result


def main():
    default = [1.0, 2.0, 3.0]

    gauss1 = Gauss1().execute()
    gauss2 = Gauss2().execute()

    print('Гаусс: ', gauss1)
    print('Гаусс c выбором главного элемента: ', gauss2)
    print('Считаем погрешность: ', '\n',
          error(default, gauss1), '\n',
          error(default, gauss2), '\n')
    input('Нажмите Enter для выхода')


if __name__ == "__main__":
    main()
