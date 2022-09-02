class Car():
	def __init__(self,year, model,item =0):
		self.year = year
		self.model = model
		self.item = 0
	def print_test(self):
		Car.__test__(self)
	def build(self,item):
		print(self.item)

	def __test__(self):
		print(f"the car builed in {self.year}")

	def __pm__(self):
		print(f"the car mode is {self.model}")


class NewCar(Car):
	def __init__(self,year, model,item = 0,value = 0):
		super().__init__(year, model,item)
		self.value = value
	def test(self):
		Car.__test__(self)
	def superbuild(self,item):
		print(self.item)

mycar = Car(2019,"tesla")
mycar.print_test()

print(mycar.model)
newcar = NewCar(2018,"tesla",6000)
newcar.test()

newcar.superbuild(0)