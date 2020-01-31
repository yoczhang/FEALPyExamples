class Person(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.weight = 'weight'

    def talk(self):
        print("person is talking....")


class Chinese(Person):
    def __init__(self, name, language):  # 先继承，在重构
        age = 33
        super(Chinese, self).__init__(name, age)  # 继承父类的构造方法，也可以写成：super(Chinese,self).__init__(name,age)
        self.language = language  # 定义类的本身属性

    def person_info(self):
        print('Person information: ')
        print('name: ', self.name)
        print('age: ', self.age)
        print('language: ', self.language)


c = Chinese('AAA', 'chinese')
c.person_info()
c.talk()
