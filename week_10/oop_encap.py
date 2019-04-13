import pdb
class Demo_1:
    x = 0 
    def __init__(self, i):
        self.i = i
        Demo_1.x += 1 

class Demo_2:
    __x = 0
    def __init__(self,i):
        self.__i = i
        Demo_2.__x += 1

class Demo_3:
    __x = 0
    def __init__(self,i):
        self.__i = i
        Demo_3.__x += 1
    @classmethod
    def get_x(cls):
        return cls.__x
    def get_i(self):
        return self.__i

if __name__ == '__main__':
    pdb.set_trace()
