def get_format_naem(first,last,middle = ''):
	if middle:
		full_name = f"{first} {middle} {last}"
	else:
		full_name = f"{first} {last}"
	return full_name.title()

import unittest

class NameTestCase(unittest.TestCase):
	def  test_fist_last_name(self):
		format_name  = get_format_naem('jains','joplin')
		self.assertEqual(format_name,'Jains Joplin')
	def test_first_last_middle_name(self):
		formatted_name = get_format_naem('wolfgang','mozart','amadeus')
		self.assertEqual(formatted_name,'Wolfgang Amadeus Mozart')

if __name__ =="__main__":
	unittest.main()
