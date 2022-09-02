class Car():
	def __init__(self,year, model):
		self.year = year
		self.model = model
	def print_test(self):
		Car.__test__(self)

	def __test__(self):
		print(f"the car builed in {self.year}")

	def __pm__(self):
		print(f"the car mode is {self.model}")


class NewCar(Car):
	def __init__(self,year, model,value):
		super().__init__(year, model)
		self.value = value
	def test(self):
		Car.__test__(self)

mycar = Car(2019,"tesla")
mycar.print_test()
newcar = NewCar(2018,"tesla",6000)
newcar.test()